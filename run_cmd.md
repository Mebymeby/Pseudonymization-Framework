0. 前缀 cmd

```shell
CUDA_VISIBLE_DEVICES=0 python main.py
```

1. 只执行基准评估

```shell
--dataset xsum --data_split test --comment xsum-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache --eval_model Qwen2.5-1.5B-Instruct
--dataset xsum --data_split test --comment xsum-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache --eval_model Qwen2.5-1.5B-Instruct
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache --eval_model Qwen2.5-1.5B-Instruct
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache --eval_model Qwen2.5-1.5B-Instruct
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ori_eval --base_run --run_step 0 0 0 0 --run_eval 1 0 --record_cache
```

2. 只执行第一步: Detect 隐私实体

```shell
# xsum 摘要
--dataset xsum --data_split test --comment xsum-ner --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner
--dataset xsum --data_split test --comment xsum-prompt --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method prompt
--dataset xsum --data_split test --comment xsum-tag_mask --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mask_model --tag_model_type tag_mask --detect_method tag_mask --generate_early_stop
--dataset xsum --data_split test --comment xsum-tag_mark --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mark_model --tag_model_type tag_mark --detect_method tag_mark --generate_early_stop

# glue-mnli 分类
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-prompt --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method prompt
## 不执行
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-tag_mask --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mask_model --tag_model_type tag_mask --detect_method tag_mask --generate_early_stop
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-tag_mark --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mark_model --tag_model_type tag_mark --detect_method tag_mark --generate_early_stop

# wmt14-de-en 翻译
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-prompt --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method prompt
## 不执行
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-tag_mask --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mask_model --tag_model_type tag_mask --detect_method tag_mask --generate_early_stop
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-tag_mark --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mark_model --tag_model_type tag_mark --detect_method tag_mark --generate_early_stop

# cnn_dailymail 续写
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-prompt --run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method prompt
## 不执行
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-tag_mask --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mask_model --tag_model_type tag_mask --detect_method tag_mask --generate_early_stop
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-tag_mark --run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mark_model --tag_model_type tag_mark --detect_method tag_mark --generate_early_stop
```

3. 只执行第二步: Generate 候选实体 [Detect 方法基于 ner]

```shell
# xsum
--dataset xsum --data_split test --comment xsum-ner-rand --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand
--dataset xsum --data_split test --comment xsum-ner-prompt --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method prompt --llm_generate_entity_prompt_input entity_and_type

# tag_mask 适配 top_k [pre 30 done] [经验证，废弃 topK 方案]
--dataset xsum --data_split test --comment xsum-tag_mask-top_k --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --tag_model tag_mask_model --tag_model_type tag_mask --detect_method tag_mask --generate_method top_k --generate_early_stop

# glue-mnli
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-prompt --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method prompt --llm_generate_entity_prompt_input entity_and_type

# wmt14-de-en
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-prompt --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method prompt --llm_generate_entity_prompt_input entity_and_type

# cnn_dailymail
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-prompt --run_step 1 1 0 0 --fast_run_step 1 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method prompt --llm_generate_entity_prompt_input entity_and_type
```

4. 只执行第三步: Replace 隐私实体 [Detect 方法基于 ner] [Generate 方法基于 rand]

```shell
# xsum
--dataset xsum --data_split test --comment xsum-ner-rand-direct --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset xsum --data_split test --comment xsum-ner-rand-prompt --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset xsum --data_split test --comment xsum-ner-rand-gen --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method gen --generate_early_stop

## 验证不同 generate 方式对困惑度的影响
--dataset xsum --data_split test --comment xsum-ner-prompt-direct --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method prompt --llm_generate_entity_prompt_input entity_and_type --replace_method direct

# glue-mnli
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-direct --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-prompt --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-gen --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method gen --generate_early_stop

# wmt14-de-en
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-direct --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-prompt --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-gen --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method gen --generate_early_stop

# cnn_dailymail
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-direct --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-prompt --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-gen --run_step 1 1 1 0 --fast_run_step 1 1 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method gen --generate_early_stop
```

5. 只执行假名化评估

```shell
# xsum
--dataset xsum --data_split test --comment xsum-ner-rand-direct-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset xsum --data_split test --comment xsum-ner-rand-prompt-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset xsum --data_split test --comment xsum-ner-rand-gen-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen

# glue-mnli
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-direct-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-prompt-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-gen-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen

# wmt14-de-en
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-direct-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-prompt-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-gen-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen

# cnn_dailymail
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-direct-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-prompt-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method prompt
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-gen-pseu_eval --run_step 1 1 1 0 --fast_run_step 1 1 1 0 --run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen
```

6. 只执行第四步: Recover 假名化输出

```shell
# xsum
--dataset xsum --data_split test --comment xsum-ner-rand-direct-direct --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset xsum --data_split test --comment xsum-ner-rand-gen-gen --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# glue-mnli
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-direct-direct --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-gen-gen --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# wmt14-de-en
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-direct-direct --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-gen-gen --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# cnn_dailymail
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-direct-direct --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-gen-gen --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 0 1 --fast_run_eval 0 1 --record_cache --detect_method ner --generate_method rand --replace_method gen --recover_method gen
```

7. 只执行结果评估

```shell
# xsum
--dataset xsum --data_split test --comment xsum-ner-rand-direct-direct-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset xsum --data_split test --comment xsum-ner-rand-gen-gen-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# glue-mnli
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-direct-direct-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset glue-mnli --data_split validation_matched --comment glue-mnli-ner-rand-gen-gen-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# wmt14-de-en
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-direct-direct-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset wmt14-de-en --data_split test --comment wmt14-de-en-ner-rand-gen-gen-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method gen --recover_method gen

# cnn_dailymail
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-direct-direct-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method direct --recover_method direct
--dataset cnn_dailymail --data_split test --comment cnn_dailymail-ner-rand-gen-gen-result --run_step 1 0 0 1 --fast_run_step 1 0 0 1 --run_eval 1 1 --fast_run_eval 1 1 --detect_method ner --generate_method rand --replace_method gen --recover_method gen
```

8. Train 数据准备

```shell
--dataset xsum --data_split train --data_size 5000 --comment xsum-ner-rand-direct --run_step 1 1 1 0 --fast_run_step 0 0 0 0 --run_eval 0 0 --record_cache --detect_method ner --generate_method rand --replace_method direct
```

9. End2End Attempt

```shell
--dataset xsum --data_split test --data_size 5000 --comment xsum-ner-rand-direct-direct --run_step 1 1 1 1 --fast_run_step 1 1 1 0 --run_eval 1 1 --detect_method ner --generate_method rand --replace_method direct --recover_method direct --end2end
```