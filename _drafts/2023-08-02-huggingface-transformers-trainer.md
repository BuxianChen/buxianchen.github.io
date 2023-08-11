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

Trainer 的所有方法如下:

```python
__init__

create_accelerator_and_postprocess

# ============ callback, state, control ==================
add_callback
pop_callback
remove_callback
call_model_init

# ======= training(train_dataset) ============
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

_maybe_log_save_evaluate
_tune_save_checkpoint

# ============ evaluate(eval_dataset) =============
evaluate

# ============ predict(test_dataset) ================
predict

# predict与evaluate都可能会调用evaluation_loop或prediction_loop,这两种“loop”最终都触发prediction_step
# 默认 use_legacy_prediction_loop 为 False, 此时 evaluate 和 predict 都走 evaluation_loop

evaluation_loop
prediction_loop
prediction_step


# =========== train/evalauate/predict ========
_get_eval_sampler
_get_train_sampler
get_test_dataloader
get_train_dataloader
get_eval_dataloader
_remove_unused_columns
_get_collator_with_removed_columns
_set_signature_columns_if_needed
_gather_and_numpify

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


_get_output_dir
_hp_search_setup


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

## 样例

```python
# trainer.train 的输出:
class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    metrics: Dict[str, float]

# trainer.prediction_loop/trainer.evaluation_loop 的输出:
class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
# trainer.evaluate 的输出是 EvalLoopOutput.metrics

# trainer.predict 的输出
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
    tokenizer
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

以下对这个例子做分析

## Trainer.train/evaluate/predict

Trainer.train 的总体逻辑:
- train 实际上就是 _inner_training_loop, 完成了整个(多个epoch)的训练过程, 真正干活的是: training_step 与 compute_loss (可以重载).
- train 的 hook 的调用点有几项: on_train_begin, on_epoch_begin, on_step_begin, on_step_end/on_substep_end, on_epoch_end, on_train_end
- train的默认hook有如下:

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

- evaluate/predict 实际上就是 evaluation_loop, 完成了整个评估, 真正干活的是: prediction_step
- evaluation_loop 与 prediction_loop 似乎有一定差别
- evaluate 与 predict 基本上是一样的 (但返回结果predict信息更多,evaluate更少)

注记:
- use_legacy_prediction_loop 默认为 False, 此时 evaluate 和 predict 都走到 evaluation_loop
- evaluation_loop 与 predict_loop 的区别: 不确定


## Trainer 修改指南

## 案例分析 1: run_glue.py

## 案例分析 2: chatglm2