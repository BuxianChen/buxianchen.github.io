---
layout: post
title: "(WIP) Stanford Alpaca"
date: 2023-10-31 11:10:04 +0800
labels: [paper, llm]
---


## 微调数据生成


总体逻辑如下, 首先研究人员手写了一些种子数据 (一共 175 个), 每条数据包含指令以及期望的输入输出结果, 然后将这些种子数据作为示例, 跟 ChatGPT (严格地说, 实现里是调用了 OpenAI 的 instruct-gpt 的一个版本 `text-davinci-003`), 说要生成更多指令及输入输出结果. 由此得到微调用的指令数据. 得到数据之后, 在 LLama 上进行微调.


以下是一些具体的细节:

- 输入给 ChatGPT 要新的指令数据的 few-shot 样例总是从种子数据中选取
- 得到 ChatGPT 输出的指令数据后, 使用 RougeScorer 计算它和种子数据以及已经生成的指令的相似度 (仅对指令做相似度计算, 不涉及期望输入与期望输出), 如果存在已有相似度较高 (大于 0.7) 的指令, 则丢弃这条新生成的指令

种子数据的一个示例如下:

```json
{
    "id": "seed_task_0",
    "name": "breakfast_suggestion",
    "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?",
    "instances": [
        {
            "input": "",
            "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."
        }
    ],
    "is_classification": false
}
```

简单翻译一下 (By gpt-3.5-turbo API 😄)

```json
{
    "instruction": "有没有早餐的选择，既不包含鸡蛋，又有蛋白质，并且大约含有700-1000卡路里的热量？",
    "input": "",
    "output": "是的，你可以选择1杯燕麦香蕉蛋白奶昔和4片培根。燕麦香蕉蛋白奶昔包含1/2杯燕麦片，60克乳清蛋白粉，1/2根香蕉，1汤匙亚麻籽油和1/2杯水，总共约550卡路里。4片培根约含有200卡路里。"
}
```


共同的 `prompt` 如下 (保存在 `prompt.txt` 中, 这段话中其实序号有点小问题):

```
You are asked to come up with a set of 20 diverse task instructions. These task instructions will be given to a GPT model and we will evaluate the GPT model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
2. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in English.
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of 20 tasks:
```

简单翻译一下 (By ChatGPT 😄)

```
你被要求提供一组多样化的20个任务指令。这些任务指令将提供给一个GPT模型，我们将评估GPT模型完成这些指令的表现。

以下是要求：

1. 尽量避免在每个指令中重复使用动词，以最大程度地增加多样性。
2. 指令使用的语言也应多样化。例如，您应该将问题与祈使句结合使用。
3. 指令类型应多样化。应包括各种类型的任务，如开放式生成、分类、编辑等。
4. GPT语言模型应能够完成指令。例如，不要要求助手创建任何视觉或音频输出。另一个例子，不要要求助手在下午5点叫醒您或设置提醒，因为它无法执行任何动作。
5. 指令应为英语。
6. 指令应为1到2句话长。允许使用祈使句或问句。
7. 您应生成适用于指令的适当输入。输入字段应包含为指令提供的具体示例。它应涉及现实数据，不应包含简单的占位符。输入应提供丰富的内容，以使指令具有挑战性，但最好不要超过100个字。
8. 并非所有指令都需要输入。例如，当一个指令询问一些常规信息，比如“世界上最高的山峰是什么”，无需提供具体的上下文。在这种情况下，我们只需在输入字段中放置"<noinput>"。
9. 输出应为指令和输入的适当响应。确保输出不超过100个字。
```


对 OpenAI API 接口调用的完整 prompt 如下:

```
<prompt>
###
1. Instruction: <instuction_1>
1. Input:
{input_1}
1. Output:
{output_1}
###
2. Instruction: <instuction_2>
2. Input:
{input_2}
2. Output:
{output_2}
###
3. Instruction: <instuction_3>
3. Input:
{input_3}
3. Output:
{output_3}
###
4. Instuction:
```