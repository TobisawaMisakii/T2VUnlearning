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

* 把hugging face pipeline 的 \_\_call\_\_ 方法上的 *@torch.no_grad()* 删了，但是尝试过后发现4090的24g显存完全不够用，如果全参数微调的话，需要47~65GB显存，租了一个H20-NVLink-80G的试试。如果带梯度图的推理都完不成，那是不是说明就算换成“手动实现的cogvideox”架构，从随机latent开始推这个transformer，其实也会爆？（后续在4090上，把pipeline拆开手动实现inference，发现一旦进入transformer就会立刻爆显存？h20没爆）
* 只把transformer块拿出来？我知道transformer的结构，可以反推他的输入的形状，然后拿进去训练？不对，我还是得从两个prompt开始，transformer及transformer以前的都得保留，一直到transformer输出预测的 v ，之后的结构倒是不需要。是不是不用step？训练流程到底是啥样的？



## 手动inference

根据pipeline的源码，手动实现一个无`Classifier-Free-Guidance`的inference，代码在：`T2VUnlearning/unlearn_train_cogvideox_decomposed.py`下的`unlearn_train()`方法，可用其同名sh脚本允许调参，设置参数`is_train`则进入unlearning training，不设置参数`is_train`可以测试手动实现的推理是否正确（进入手搓的inference）

对transformer等其他主干进行冻结，加入eraser之后的adapter_transformer只允许adapter的参数使用梯度，模型能在h20上正常训练（如果加入Classifier-Free Guidance，prompt embedding变为之前的二倍大小，就算是h20的96G显存还是会爆）。

```shell
CUDA_VISIBLE_DEVICES=0 python unlearn_train_cogvideox_decomposed.py \
  --is_train True\
  --concept="nudity" \
  --prompt_path="/root/autodl-tmp/T2VUnlearning/evaluation/data/10prompt_pairs.csv" \
  --model_path /root/autodl-tmp/models/cogvideox-2b \
  --eraser_rank 128 \
  --eraser_ckpt_path="/root/autodl-tmp/T2VUnlearning/adapter/10prompt_50step_1epoch_cogvideox2b_nudity_erasure" \
  --num_inference_steps 50 \
  --num_frames 17 \
  --num_epoch 1 \
  --generate_type t2v \
  --dtype float32 \
  --eta 7.0 \
  --alpha 1.0 \
  --beta 0.0 \
  --seed 42
```

超参：

* $\alpha$ 是localization loss的系数
* $\beta$ 是preservation loss的系数
* $\eta$ 为$L_{unlearn}$中 $v_{neg}$ 所需参数

取值均参考 T2VUnlearning原文

对transformer等其他主干进行冻结，加入eraser之后的adapter_transformer只允许adapter的参数使用梯度，模型能在h20上正常训练（如果加入Classifier-Free Guidance，prompt embedding变为之前的二倍大小，就算是h20的96G显存还是会爆）。

注意，在训练时我设定了 $d_{type} = float32$ ，如果想采用float16为模型参数，需要在 adamw 优化adapter参数之前，将adapter参数全部转为float32才不会爆nan错误（可能有其他方法，但设定float32可无脑解决此问题） 

## Debug

```shell
  0%|                                     | 0/50 [00:00<?, ?it/s]
v_unsafe_adapter tensor(-0.0489, device='cuda:0', grad_fn=<MeanBackward0>) tensor(1.1342, device='cuda:0', grad_fn=<StdBackward0>)
v_unsafe_origin tensor(0.0464, device='cuda:0') tensor(1.0773, device='cuda:0')
True <MeanBackward0 object at 0x7f68f0d92e90>
Epoch 0, Step 1/50, Loss: 2.2981321811676025

  2%|██                                   | 1/50 [00:11<09:41, 11.87s/it]
v_unsafe_adapter tensor(nan, device='cuda:0', grad_fn=<MeanBackward0>) tensor(nan, device='cuda:0', grad_fn=<StdBackward0>)
v_unsafe_origin tensor(-0.0166, device='cuda:0') tensor(1.0902, device='cuda:0')
True <MeanBackward0 object at 0x7f68f0d92ec0>
Epoch 0, Step 2/50, Loss: nan
```

从第二步去噪开始，`v_unsafe_adapter`变成`NaN`，而 `v_unsafe_origin` 一切正常，考虑是eraser初始化的问题，对`AdapterEraser`在其`__init__`方法中进行正态初始化，再尝试训练，仍然第二步就爆掉...
并且发现原始代码对`AdapterEraser`的up层的初始化是`nn.init.zeros_`，并且在`CogVideoXWithEraser`中是使用残差连接的，说明可能并不需要初始化，直接设置为0即可


```shell
  0%|               | 0/50 [00:00<?, ?it/s]
loss_unlearn: 2.7803542613983154
v_neg mean/std: -0.5395044088363647 1.9355318546295166
v_unsafe_adapter mean/std: 0.04594830796122551 1.0853095054626465
Epoch 0, Step 1/50, Loss: 2.7804088592529297
  2%|██             | 1/50 [00:11<09:40, 11.85s/it]
loss_unlearn: nan
v_neg mean/std: 0.15456829965114594 1.727604627609253
v_unsafe_adapter mean/std: nan nan
Epoch 0, Step 2/50, Loss: nan
```

优化器前进一步，adapter参数直接被冲飞

```shell
transformer_blocks.0.attn1.adapter.down.weight - mean: nan, std: nan, requires_grad: True
transformer_blocks.0.attn1.adapter.down.bias - mean: nan, std: nan, requires_grad: True
transformer_blocks.0.attn1.adapter.up.weight - mean: nan, std: nan, requires_grad: True
transformer_blocks.0.attn1.adapter.up.bias - mean: nan, std: nan, requires_grad: True
......
```

下午试试加上grad clip？（在设定模型参数为`float16`时，adapter参数更新时使用了grad clip）

```shell
[DEBUG] step=999
v_noprompt_origin mean/std: -0.059468552470207214 1.0926387310028076
v_unsafe_origin   mean/std: 0.016997864469885826 1.0744503736495972
v_neg             mean/std/max/min: -0.135934978723526 1.1422020196914673 2.5693359375 -2.515869140625
v_unsafe_adapter  mean/std/max/min: -0.048881810158491135 1.1342509984970093 1.8837890625 -1.7705078125
loss_unlearn: 0.09701912105083466
loss_preserve: 0.034152764827013016
total loss: 0.09701912105083466
Epoch 0, Step 1/50, Loss: 0.09701912105083466
[DEBUG] step=979
v_noprompt_origin mean/std: -0.17507442831993103 1.0267356634140015
v_unsafe_origin   mean/std: -0.13716070353984833 0.996391773223877
v_neg             mean/std/max/min: -0.21298815310001373 1.0671366453170776 1.62890625 -3.259765625
v_unsafe_adapter  mean/std/max/min: nan nan nan nan
loss_unlearn: nan
loss_preserve: nan
total loss: nan
[WARNING] v_unsafe_adapter contains NaN or Inf at step 979
[WARNING] loss contains NaN or Inf at step 979
Epoch 0, Step 2/50, Loss: nan
[DEBUG] step=959
v_noprompt_origin mean/std: -0.11281387507915497 0.9431551098823547
v_unsafe_origin   mean/std: -0.08958429098129272 0.923428475856781
v_neg             mean/std/max/min: -0.13604344427585602 0.969879686832428 1.501953125 -2.3505859375
v_unsafe_adapter  mean/std/max/min: nan nan nan nan
loss_unlearn: nan
loss_preserve: nan
total loss: nan
[WARNING] v_unsafe_adapter contains NaN or Inf at step 959
[WARNING] loss contains NaN or Inf at step 959
```

**Step 1**：loss 正常，`v_neg` 和 `v_unsafe_adapter` 都是健康的范围（±2.5）。

**Step 1 梯度**：几乎所有 adapter 的 `down.weight`、`up.weight` 梯度都是 **0**，只有 `up.bias` 有非常小的梯度（~1e-4）。

**Step 2**：`v_unsafe_adapter` 直接 NaN。

**Step 2 梯度**：全是 NaN。



把adam更新的参数变为float32，不再是NaN，新问题是：

```shell
[DEBUG] step=979
v_noprompt_origin mean/std: -0.002673887647688389 0.9902946352958679
v_unsafe_origin   mean/std: 0.03506551310420036 0.9863004684448242
v_neg             mean/std/max/min: -0.26684969663619995 1.311755895614624 3.1474609375 -3.44677734375
v_unsafe_adapter  mean/std/max/min: -0.04738422855734825 1.1296082735061646 1.8876953125 -1.75390625
loss_unlearn: 0.7120194435119629
total loss: 0.7120194435119629
Epoch 0, Step 2/50, Loss: 0.7120194435119629
[transformer_blocks.0.attn1.adapter] down.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.0.attn1.adapter] down.bias grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.0.attn1.adapter] up.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.0.attn1.adapter] up.bias grad mean/std/max: 7.927417755126953e-06 0.0002830028533935547 0.006542205810546875
[transformer_blocks.1.attn1.adapter] down.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.1.attn1.adapter] down.bias grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.1.attn1.adapter] up.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.1.attn1.adapter] up.bias grad mean/std/max: 2.4437904357910156e-06 0.00024330615997314453 0.004932403564453125
[transformer_blocks.2.attn1.adapter] down.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.2.attn1.adapter] down.bias grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.2.attn1.adapter] up.weight grad mean/std/max: 0.0 0.0 0.0
[transformer_blocks.2.attn1.adapter] up.bias grad mean/std/max: 3.7550926208496094e-06 0.00017023086547851562 0.00411224365234375
```

只有up.bias是能更新的

发现：

```python
class CogVideoXWithEraser(nn.Module):
    def __init__(
        self,
        attn,
        eraser_rank
    ):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.to_v.weight.shape[-1], eraser_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        print("[DEBUG attn input] mean:", hidden_states.mean().item(), "std:", hidden_states.std().item())
        hidden_states, encoder_hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            **cross_attention_kwargs,
        )
        print("[DEBUG attn output] mean:", hidden_states.mean().item(), "std:", hidden_states.std().item())

        if self.adapter.use_eraser:
            hidden_states = hidden_states + self.adapter(hidden_states)

        return hidden_states, encoder_hidden_states
```

进入attn之前数据还正常，

```shell
[DEBUG attn input] mean: 0.01464080810546875 std: 1.3896484375
[DEBUG attn input] mean: 0.0032024383544921875 std: 1.4443359375
[DEBUG attn input] mean: 0.01358795166015625 std: 1.529296875
[DEBUG attn input] mean: 0.0024738311767578125 std: 1.4873046875
[DEBUG attn input] mean: -0.00496673583984375 std: 1.1435546875
[DEBUG attn input] mean: -0.01428985595703125 std: 1.3095703125
[DEBUG attn input] mean: -0.0022735595703125 std: 1.171875
```

从attn出来之后，全部变为0

```shell
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
[DEBUG attn output] mean: 0.0 std: 0.0
```

```shell
to_q.weight std: 0.0
to_k.weight std: 0.0
to_v.weight std: 0.0
to_q.weight std: 0.0
to_k.weight std: 0.0
to_v.weight std: 0.0
```

原来是set up erasers时出错：

```python
def setup_cogvideo_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, CogVideoXBlock):
                print("changing: ",name)
                original_attention = module.attn1
                modified_attention = CogVideoXWithEraser(original_attention, eraser_rank).to(dtype=dtype)
                # modified_attention.to_empty(device=device)
                modified_attention.to(device=device)
                module.attn1 = modified_attention
```

之前的to_empty()会导致attn层全部重置为0

修改之后成功训练



现在已经能根据unlearning loss训练并保存eraser了

`T2VUnlearning/adapter/self_cogvideox2b_nudity_erasure`：在`ring-a-bell`前十对prompt上，一个prompt去噪推理2步17frames，1个epoch（推理时加入这个eraser几乎没有任何改变）

`T2VUnlearning/adapter/10prompt_50step_1epoch_cogvideox2b_nudity_erasure`：在`ring-a-bell`前十对prompt上，一个prompt去噪推理50步17frames，1个epoch

在这个eraser上测试（`test_cogvideo.sh`设定eraser的ckpt即可）
clean:

![image-20250725205601112](README.assets/image-20250725205601112-1753448166394-1.png)

erased:

![image-20250725205626409](README.assets/image-20250725205626409.png)

因为实际上目前只用了第一个loss $L_{unlearn}$，对保持概念的保护并不好，但基本复现走通了训练流程
