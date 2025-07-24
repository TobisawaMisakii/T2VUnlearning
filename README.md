# T2VUnlearning

整体思路1（已废弃）：

* 北大仓库提供的带adapter的架构
* 直接用diffusers的CogVideoX pipeline
* 使用hook从pipeline中抓取需要的中间输出

整体思路2：

* 北大仓库提供的带adapter的架构
* 仿照diffusers库，实现一个CogVideoX的inference

## Adapter

## adapter structure

`T2VUnlearning/receler/erasers/utils.py`

```python
class AdapterEraser(nn.Module, EraserControlMixin):
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.down = nn.Linear(dim, mid_dim)
        self.act = nn.GELU()
        self.up = zero_module(nn.Linear(mid_dim, dim))

    def forward(self, hidden_states):
        return self.up(self.act(self.down(hidden_states)))
```

对于cogvideox-2b:

``````
AdapterEraser(
  (down): Linear(in_features=1920, out_features=128, bias=True)
  (act): GELU(approximate='none')
  (up): Linear(in_features=128, out_features=1920, bias=True)
)
``````

## insert adapter to transformer

先把插入adapter后的自定义transformer层`CogVideoXWithEraser`

`T2VUnlearning/receler/erasers/cogvideo_erasers.py`

```python
class CogVideoXWithEraser(nn.Module):
    def __init__(
        self,
        attn,
        eraser_rank
    ):
        ......

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
		......
        return hidden_states, encoder_hidden_states
```

在训练/推理时，用自定义的transformer块替换原始模型，即用`CogVideoXWithEraser`替换`CogVideoXBlock`

### train

提供了 **设定新的erasers，修改transformer块，命名adapter** 的方法

`T2VUnlearning/receler/erasers/cogvideo_erasers.py`

```python
def setup_cogvideo_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, CogVideoXBlock):
                print("changing: ",name)
                original_attention = module.attn1
                modified_attention = CogVideoXWithEraser(original_attention, eraser_rank).to(device = device, dtype = dtype)
                module.attn1 = modified_attention

    replace_transformer_block(model)
    erasers = {}
    for name, module in model.named_modules():
        if isinstance(module, CogVideoXWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers
```

在cogvideox-2b下：

Number of modules in eraser: 30 （因为有30层DiT）

Number of parameters in eraser: 14807040

 每层：

* 权重：`1920 × 128 = 245760`

* bias：`128`

总计：`245760 + 128 = 245888` 个参数

将adapter参数传入优化器即可训练，最好打开`pipe.enable_sequential_cpu_offload()`优化

### inference

采用保存的eraser weights，使用 `inject_eraser` 到transformer

`T2VUnlearning/receler/erasers/cogvideo_erasers.py`

```python
def inject_eraser(transformer, eraser_ckpt, eraser_rank, eraser_type='adapter'):
    for name, module in transformer.named_modules():
        if isinstance(module, CogVideoXBlock):
            print("changing: ",name)
            original_attention = module.attn1
            modified_attention = CogVideoXWithEraser(original_attention, eraser_rank)
            module.attn1 = modified_attention
            eraser_name = f'{name}.attn1.{eraser_type}'
            module.attn1.adapter.load_state_dict(eraser_ckpt[eraser_name])
            module.attn1.adapter.to(device = transformer.device, dtype = transformer.dtype)
```

## Loss （cogvideo-2b为例）

### $L_{unlearn}$

需要拿到的是：

* **adapter**模型， 在**unsafe prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta'}(x_t, c, t)$
* **non-adapter**模型，在**unsafe prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta}(x_t, c, t)$
* **non-adapter**模型，在**empty prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta}(x_t, t)$ （？文中似乎并未明说是empty prompt ？有待商榷）

$$
v_{neg} = v_θ(x_t, t) − η(v_θ(x_t, c, t) − v_θ(x_t, t)),
$$

$$
L_{unlearn} = E_{x,c,ε,t} ∥v_{neg} − v_{θ′} (x_t, c, t)∥_2^2.
$$

### $L_{preserve}$

需要拿到的是：

* **adapter**模型， 在**safe prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta'}^{pre}$
* **non-adapter**模型，在**safe prompt**下的第30个（最后一个）transformer block的 feedforward layer 输出 $v_{\theta}^{pre}$

$$
L_{pre} =  E_{x^{pre},c^{pre},ε,t} ∥v_{θ′} (x_t, c^{pre}, t) − v_{θ} (x_t, c^{pre}, t)∥_2^2.
$$

### $L_{loc}$

借鉴了 `receler`：<https://github.com/jasper0314-huang/Receler> 的 Concept-Localized Regularization for Erasing Locality

TODO



## Hook（discarded）

### Method

因为hugging face的pipeline实现非常集成化，并且其 **\_\_call\_\_** 方法封装很好，所以没有采用“用部件搭出cogvideox pipeline并拿到中间变量”的方法，而是转向 hook 这种方法

在 PyTorch 中，**Hook 是一种在前向传播或反向传播过程中，动态获取或修改模块中间结果的机制**，常用于：

- 捕获中间层输出或输入
- 实现可视化、调试、特征分析
- 插入自定义行为，不改动模型结构

**Forward hook**：注册在模块上，用于捕获前向传播的输入/输出。

```python
handle = module.register_forward_hook(hook_fn)
```

使用流程：

```python
intermediate_outputs = {}

def hook_fn(module, input, output):
    intermediate_outputs['layer_name'] = output

model.layer.register_forward_hook(hook_fn)
```

`input` 是一个 tuple，表示模块的输入

`output` 是模块的输出，通常是 Tensor 或 tuple

可以在 hook 中把结果保存下来，供后续使用

### Experiment

在实验中我使用了 **forward hook**，从模型内部的特定模块（如 transformer 的某一层）提取特征（例如最后一层 FFN 的输出）

```python
target_module_names = [
    "transformer_blocks.29.ff",
]
```

拿到的tensor形状是： With adapter transformer_blocks.29.ff: **torch.Size([2, 6976, 1920])**

batch-size为2是因为开启了classifier-free guidance （默认的超参 guidance_scale = 6.0）：

```python
do_classifier_free_guidance = guidance_scale > 1.0

# 3. Encode input prompt
prompt_embeds, negative_prompt_embeds = self.encode_prompt(
	......
)

if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
```

可以取出prompt embeds对应的 $v_{\theta'}(x_t, c, t)$

```python
v_unsafe_adapter = intermediate_outputs_with_adapter_unsafe[target_module_names[0]][1]
```

同理，可以拿到六个向量：

* v_unsafe_adapter :  $v_{\theta'}(x_t, c, t)$
* v_unsafe_origin：$v_{\theta}(x_t, c, t)$
* v_safe_adapter：$v_{\theta'}(x_t, c^{pre}, t)$
* v_safe_origin：$v_{\theta}(x_t, c^{pre}, t)$
* v_noprompt_origin ： $v_{\theta}(x_t, t)$

形状均为：torch.Size([6976, 1920])

随后即可计算$L_{unlearn}$和$L_{preserve}$

### Problem: grad

问题：hook从pipeline中抓取出来的张量没有grad！

可能的办法：

* 把hugging face pipeline 的 \_\_call\_\_ 方法上的 *@torch.no_grad()* 删了，但是尝试过后发现4090的24g显存完全不够用，如果全参数微调的话，需要47~65GB显存，租了一个H20-NVLink-80G的试试。如果带梯度图的推理都完不成，那是不是说明就算换成“手动实现的cogvideox”架构，从随机latent开始推这个transformer，其实也会爆？（后续在4090上，把pipeline拆开手动实现inference，发现一旦进入transformer就会立刻爆显存？==h20没爆==）
* 只把transformer块拿出来？我知道transformer的结构，可以反推他的输入的形状，然后拿进去训练？不对，我还是得从两个prompt开始，transformer及transformer以前的都得保留，一直到transformer输出预测的 v ，之后的结构倒是不需要。是不是不用step？训练流程到底是啥样的？



## 手动inference

h20还是没坚持到算完所有 v  (transformer的输出)

但是只用一个transformer，预测一下噪声这样的简单inference工作已经完成，说明流程上已经接近完成

```shell
CUDA_VISIBLE_DEVICES=0 python unlearn_train_cogvideox_transformer.py \
  --concept="nudity" \
  --prompt_path="/root/autodl-tmp/T2VUnlearning/evaluation/data/nudity-ring-a-bell.csv" \
  --model_path /root/autodl-tmp/models/cogvideox-2b \
  --eraser_rank 128 \
  --eraser_ckpt_path="/root/autodl-tmp/T2VUnlearning/adapter/self_cogvideox2b_nudity_erasure" \
  --num_inference_steps 50 \
  --num_frames 49 \
  --num_epoch 2 \
  --generate_type t2v \
  --dtype float16 \
  --eta 7.0 \
  --seed 42
```

用此脚本即可用作 unlearning 训练 （目前已实现 $L_{unlearn}$ 和 $L_{preserve}$ ）