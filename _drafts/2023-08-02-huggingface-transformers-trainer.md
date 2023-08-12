---
layout: post
title: "(WIP) 🤗 Transformers Trainer API"
date: 2023-08-02 22:00:04 +0800
labels: [huggingface]
---

## 动机、参考资料、涉及内容

动机

- 使用 🤗 Transformers 时涉及到模型训练的代码怎么写才是优雅的方式
- 🤗 Transformers Trainer 的实现逻辑

涉及内容

- 🤗 Transformers Trainer 的实现细节
- 应该怎样按需在 Trainer 的基础上修改/增加功能

## Trainer 使用参考

🤗 Transformers GitHub 项目里包含了许多端到端的例子, Trainer API 的使用可以借鉴 [examples/pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch) 底下的内容, 粗略总结如下:

```
# speed
benchmarking

# NLP
language-modeling
  - run_clm.py: Trainer
  - run_mlm.py: Trainer
  - run_plm.py: Trainer
question-answering
  - run_qa.py: QuestionAnsweringTrainer(自定义)
  - run_qa_beam_search.py: QuestionAnsweringTrainer(自定义)
  - run_seq2seq_qa.py: QuestionAnsweringSeq2SeqTrainer(自定义)
summarization
  - run_summarization.py: Seq2SeqTrainer
text-classification
  - run_glue.py: Trainer
  - run_xnli.py: Trainer
text-generation(仅含推理)
token-classification
  - run_ner.py: Trainer
translation
  - run_translation.py: Seq2SeqTrainer
multiple-choice(swag, 选择题)
  - run_swag.py: Trainer

# Audio
audio-classification
speech-pretraining
speech-recognition

# CV
contrastive-image-text
image-classification
image-pretraining
semantic-segmentation
```

## Trainer API 详解

本节针对如下特定版本对 Trainer 的 API 进行解释

```
accelerate==0.21.0
transformers==4.31.0
```

Trainer 的**全部**方法如下:

```python
__init__

create_accelerator_and_postprocess

# ============ callback, state, control ==================
add_callback
pop_callback
remove_callback
call_model_init

# ======= train(train_dataset) ============
train
_inner_training_loop
training_step

compute_loss
compute_loss_context_manager
autocast_smart_context_manager


_load_best_model
_load_from_checkpoint
_load_optimizer_and_scheduler
_load_rng_state
_issue_warnings_after_load


_save
_save_checkpoint
_save_tpu
save_metrics
save_model
save_state
_rotate_checkpoints
_sorted_checkpoints
_get_output_dir

_maybe_log_save_evaluate

# ============ evaluate(eval_dataset) =============
evaluate

# ============ predict(test_dataset) ================
predict

# predict与evaluate都可能会调用evaluation_loop或prediction_loop,这两种“loop”最终都触发prediction_step
# 默认 use_legacy_prediction_loop 为 False, 此时 evaluate 和 predict 都走 evaluation_loop

evaluation_loop
prediction_loop  # 源码中将这部分代码标记为 deprecated code
prediction_step


# =========== train/evalauate/predict: dataset/dataloader ========
_get_eval_sampler
_get_train_sampler
get_test_dataloader
get_train_dataloader
get_eval_dataloader
_remove_unused_columns
_get_collator_with_removed_columns
_set_signature_columns_if_needed
_gather_and_numpify

# ========== optimizer/scheduler ===============================
create_optimizer
create_optimizer_and_scheduler
create_scheduler
get_optimizer_cls_and_kwargs
_get_learning_rate

# ============= others ==========
_move_model_to_device
_nested_gather
_prepare_input
_prepare_inputs
_wrap_model
ipex_optimize_model


store_flos
is_local_process_zero
is_world_process_zero
floating_point_ops
num_examples

_hp_search_setup
_tune_save_checkpoint


_push_from_checkpoint
create_model_card
_report_to_hp_search
_add_sm_patterns_to_gitignore
init_git_repo
push_to_hub


hyperparameter_search
log
log_metrics
metrics_format
torch_jit_model_eval
```

### 样例

```python
# trainer.train 的输出:
class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]

# trainer.evaluation_loop/prediction_loop 的输出:
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
# trainer.evaluate 的输出是 EvalLoopOutput.metrics

# trainer.predict 的输出: 其实就是 EvalLoopOutput 去掉 num_samples 属性
class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]

# text-classification/run_glue.py 简化后如下
# Trainer 即可以用于训练(单卡/多卡),也可以用于验证(带标签,跟训练一致,单卡/多卡),也可以用于对一个数据集做推理(测试, 不带标签, 单卡/多卡)
trainer = Trainer(
    model,
    training_args,
    data_collator,
    train_dataset,
    eval_dataset,
    tokenizer,
    model_init=None,
    compute_metrics=compute_metrics,  # Callable
    callbacks=None,
    optimizers=None,
    preprocess_logits_for_metrics=None,  # Callable
)

# training_args.resume_from_checkpoint: Optional[str]
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
train_result: "TrainOutput" = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics

metrics: "EvalLoopOutput.metrics" = trainer.evaluate(eval_dataset=eval_dataset)

# predict_dataset = test_dataset
predict_dataset = predict_dataset.remove_columns("label")
predictions: "PredictionOutput" = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
```

以下是详细分析

### 对外 API

一般来说, `Trainer` 对外的 API 主要就是上面例子中所展示的, 首先初始化一个 `Trainer` 实例, 然后调用 `train`, `evaluate`, `predict` 即可, 针对推理来说, 使用 `evaluate` 或者 `predict` 是对一个 dataset 做推理的, 好处是它也会利用到多张卡, 而如果只想对单条数据/一个batch的数据做推理的话, 可以使用 `prediction_step`

### Trainer.train

- train 实际上就是 _inner_training_loop, 完成了整个(多个epoch)的训练过程, 真正干活的是: training_step 与 compute_loss (可以重载).
- train 的 hook 的调用点有几项: on_train_begin, on_epoch_begin, on_step_begin, on_step_end/on_substep_end, on_epoch_end, on_train_end, 注意在 `Trainer` 的语境里, step 指的是一次梯度更新, 大多数与 step 的概念都是以一次梯度更新为最小单元的, 而 substep 是指一次梯度更新所需要的梯度累积次数.
- train的默认 callback 有如下:
    ```python
    # DefaultFlowCallback
    # on_step_end:
    # on_epoch_end:

    # ProgressCallback/NotebookProgressCallback/PrinterCallback
    # on_train_begin: 初始化进度条
    # on_step_end: 进度条加1
    # on_train_end:
    # on_log:
    # on_prediction_step/on_evaluate/on_predict

    # TensorBoardCallback/WandbCallback/...
    # on_train_begin
    # on_train_end
    # on_log: 在trainer中step结束后可能会通过 _maybe_log_save_evaluate 在 DefaultFlowCallback.on_step_end 触发
    # 具体逻辑是首先 DefaultFlowCallback 在 state.global_step % args.logging_steps == 0 时将 control.should_log 设定为 True, 然后调用 trainer._maybe_log_save_evaluate 时内部会触发 trainer.log, 最终归结为 trainer.callback_handler.on_log
    ```
  注意: trainer.control.on_log 先将 control.should_log = False 再触发 callbacks 的 hook, trainer.control 很多 on_xxx 都有类似的行为. 注意: TensorBoardCallback.on_log 触发时是不检查 control.should_log 的. 注意: DefaultFlowCallback 会修改 trainer.control 和 trainer.state, 而 ProgressCallback/TensorBoardCallback 不修改 trainer.control 和 trainer.state.


### Trainer.evaluate/predict

- evaluate/predict 方法实际上就是 evaluation_loop, 完成了整个评估, 真正干活的是: prediction_step. 注意: `TrainingArguments` 中包含 `use_legacy_prediction_loop` 一项, 其默认值为 `False`, 这样会导致 evaluate/predict 进入 evaluation_loop 而非 prediction_loop, 后者被标记为 deprecated code.
- evaluate 与 predict 基本上是一样的: 因为本质上都是调用一次 `evaluation_loop`, 得到一个 `EvalLoopOutput` 数据结构, 大体逻辑如下:
```python
# trainer.trainer.evaluation_loop/prediction_loop 的输出:
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
# trainer.evaluate 的输出是 EvalLoopOutput.metrics

# trainer.predict 的输出: 其实就是 EvalLoopOutput 去掉 num_samples 属性
class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]

class Trainer:
    def predict(self, ...):
        output: "EvalLoopOutput" = self.evaluation_loop(...)
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
    def evaluate(self, )
```


## 自定义 Trainer 指南

从实现角度, Trainer 的主要魔改方式有两种: 一种是写一个类继承 Trainer, 重写某些方法, 另一种是实例化 Trainer 时加入一些 callback.

从功能角度, 我们通常需要魔改部分有这些:

- 自定义数据集以及 dataloader
- 自定义损失计算逻辑
- 自定义 optimizer 与 scheduler
- 自定义日志
- 自定义模型加载
- 自定义模型保存

其中前面 4 项在下一节讨论, 后面 2 项在最后讨论

### 继承 Trainer 并重载一些方法

继承 Trainer 这种方式, 参考官方文档 [https://huggingface.co/docs/transformers/main_classes/trainer](https://huggingface.co/docs/transformers/main_classes/trainer), 主要关注以下方法即可: 

- `get_train_dataloader`, `get_eval_dataloader`, `get_test_dataloader`: 我们简要看一下其中一个的源码
    ```python
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    ```
    这三个函数重载起来比较简单, 实际上最终只是得到一个 dataloader, 注意这里使用到的几个内部方法: `_remove_unused_columns`, `_get_collator_with_removed_columns`, `_get_train_sampler`, `_get_eval_sampler` 仅在这三个方法中被使用到, 所以如果重载时不方便操作, 可以不去调用这四个内部方法, 不会引发其他地方的逻辑问题
- `log`: 这个相对来说是比较需要重载的地方, 首先看一下相关的源码
    ```python
    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def evaluate(self, ...):
        # 在 evaluation_loop 结束之后
        # output: EvalLoopOutput
        self.log(output.metrics)
    def train(self, ...):
        # 在整个训练结束之前有一次日志记录
        self.log(metrics)
    # 以下为 _maybe_log_save_evaluate 方法的完整源码, 此方法只在train中被调用: 一共两处, 一是在每次梯度更新结束后, 二是每个训练epoch结束后被调用
    # 注意 train 函数对 evaluate 的调用都是透过 _maybe_log_save_evaluate 方法的
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            # reset tr_loss to zero
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs)
        metrics = None
        if self.control.should_evaluate:  # 注意, 此处可能触发 trainer.evaluate 的调用
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    ```
    我们从日志需求的角度来举例看应该怎么优雅地满足:
    - 需要隔几个 step 打印一次该 batch 的训练数据, 即需要使用 Tensorboard 的一些保存文本的操作, 这个可以重载 `training_step`, 再重载的方法里触发 `self.log` 的调用, 并且适当重载 `self.log` (以 Tensorboard 举例, 可能还需要在适当的地方调用 `add_text` 方法, 内置的 `TensorboardCallback` 只会使用到 `add_scalar` 功能)
    - 隔几个 step/epoch 进行一次训练集的损失: 可以通过 TrainingArguments 里的参数进行相应的设置间隔数

- `create_optimizer_and_scheduler`, `create_optimizer`, `create_scheduler`: 注意在 train 中, scheduler 的更新频率是每次梯度更新就更新一次
- `compute_loss`, `training_step`:
    ```python
    def train(self, ...):
        for epoch in range(epochs_trained, num_train_epochs):
            # ...
            for step, inputs in enumerate(epoch_iterator):
                # ...
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)
                # ...

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)  # 注意 compute_loss 在 prediction_step 中也有可能被调用
        if self.args.n_gpu > 1:
            # 这里不是很理解, 会触发多GPU之间的通讯吗?
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    ```

关于训练损失, 这里做几点说明 (以 DDP 举例):
- 首先从 `training_step` 出来的损失是一张卡上这一个batch的平均损失
- 假设每隔 10 个 step 进行一次日志打印, 每张卡上会将这 10 个 step 的损失进行加和得到 `tr_loss` (`_maybe_log_save_evaluate` 的入参之一)
- 在 `_maybe_log_save_evaluate` 内部:
    ```python
    # 假设有 4 张 GPU, 首先将 4 个 tr_loss 汇总起来求平均
    tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
    # 然后在除以 10, 得到平均损失, 总的来说这里的 logs["loss"] 是这 10 个 step 里平均到每个样本的平均损失
    logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
    ```
- `prediction_step`, `predict`, `evaluate`: Seq2SeqTrainer 主要就是重载了这三个方法, 具体可参考下一节的示例

### 例子: Seq2SeqTrainer

```python
class Seq2SeqTrainer(Trainer):
    def __init__(self, ...):
        super().__init__(self, model, args, ...)
        if self.args.generation_config is not None:
            gen_config = self.load_generation_config(self.args.generation_config)
            self.model.generation_config = gen_config
    # predict 类似, 也是同样的重载方式
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,  # 注意父类 Trainer 并不含 gen_kwargs 这个入参
    ):
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,  # 注意父类 Trainer 并不含 gen_kwargs 这个入参
    ):
        # 使用到 self._gen_kwargs
        # ...
        # 父类这里是 self.model(**inputs)
        generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        # 如果需要计算损失, 会再调用一次 self.model(**inputs)
        return loss, generated_tokens, labels
```

注意 `Seq2SeqTrainer.train` 方法沿用父类的 `Trainer.train`, 因此在 `train` 中对 `Seq2SeqTrainer.evaluate` 的调用不会传入 `gen_kwargs` 参数, 因此在训练过程里的验证步骤 (即文本生成过程里 `generate` 函数的控制参数) 依赖于实例化时的传参 `args`, 而单独调用 `evaluate` 或 `predict` 时, 可以通过传入 `gen_kwargs` 控制文本生成的参数.


### 增加 callback

至于加 callback 这种做法, 🤗 Transformers 本身内置的 callback 并不多, 实际上也足够使用了, 感觉一般也不需要再新增什么了, 包括:

```python
DefaultFlowCallback
ProgressCallback/NotebookProgressCallback/PrinterCallback
TensorBoardCallback/WandbCallback/...
# 这个需要在 Trainer 实例化时传入
EarlyStoppingCallback
```

### Trainer 的模型加载逻辑

涉及的调用关系如下, 主入口如下:

- train 函数传入 `resume_from_checkpoint` 时需要关注: `_load_from_checkpoint`, `_load_optimizer_and_scheduler`, `_load_rng_state` 即可
- Trainer 实例的参数 `args` 中设置了 `load_best_model_at_end=True` 时, 还需要关注 `_load_best_model`

```python
# 训练开始可能会加载模型, 调用 _load_from_checkpoint
_load_from_checkpoint
# 根据不同的情形, 可能会在内部触发如下:
# deepspeed_load_checkpoint: 启用 deepspeed 时
# load_sharded_checkpoint: 多个模型切片时
# load_fsdp_model: 启用 FSDP 时
# model.load_state_dict

_load_optimizer_and_scheduler
_load_rng_state
_issue_warnings_after_load

# 训练结束时, 根据初始化 Trainer 时的参数设置, 可能会加载最优的模型
_load_best_model
```

### Trainer 的模型保存逻辑

涉及的调用关系如下, 主入口如下:

- train 函数中只会透过 `_maybe_log_save_evaluate` 触发模型保存, 而它只直接触发 `_save_checkpoint` (会保存模型, 优化器状态, 随机种子等), 而保存模型的部分是由 `save_model` 来完成的, 而它根据不同的情况, 一般会透过 `_save` 来做保存.
- 在 Trainer 实例的参数 `args` 中设置了 `args.save_total_limit: int` 时, 会触发一些删除模型文件的操作, 最底层涉及到 `_rotate_checkpoints`
- train 函数在只保留一个模型文件的设定时, 还会在训练结束时做一些删除模型文件的操作 (利用 `_sorted_checkpoints`)

```python
_maybe_log_save_evaluate  # 包含了对 _save_checkpoint 的调用
_save_checkpoint  # 包含了保存权重, 优化器状态, 随机种子等
  _get_output_dir   # 被 _save_checkpoint 调用, 用于确定保存路径
  save_model        # 被 _save_checkpoint 调用, 用于保存权重, 根据不同的训练设置分别调用如下
    _save
    _save_tpu
    # save_fsdp_model
  save_metrics
  save_state

# 只在 _save_checkpoint 结束时被调用
_rotate_checkpoints
_sorted_checkpoints
```

处于调用链最低端的 `_save` 函数完整源码如下: 它需要负责保存 `model`, `tokenizer` 和训练参数 `args`

```python
def _save(self, output_dir: Optional[str] = None, state_dict=None):
    # If we are executing this function, we are the process zero, so we don't check for that.
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")

    supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(self.model, supported_classes):
        if state_dict is None:
            state_dict = self.model.state_dict()

        if isinstance(unwrap_model(self.model), supported_classes):
            unwrap_model(self.model).save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if self.args.save_safetensors:
                safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
```

## 案例分析 1: run_glue.py

## 案例分析 2: chatglm2