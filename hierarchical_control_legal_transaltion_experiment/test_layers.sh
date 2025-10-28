#!/bin/bash

# 配置环境变量
export OPENAI_EMBED_MODEL=doubao-embedding-text-240715
export OPENAI_API_KEY=258be12b-4726-448f-976b-59491ef007f2
export OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export OPENAI_API_MODEL=deepseek-v3-1-terminus

# 运行小样本测试，查看每一层是否改进翻译
python run_experiment.py \
    --test-set dataset/processed/test_set_zh_en.json \
    --samples 3 \
    --ablation full \
    --save-intermediate \
    --verbose \
    --output-dir outputs_debug \
    --max-concurrent 1

echo ""
echo "测试完成！查看上面的输出，寻找："
echo "  ⚠️  句法层未改进翻译（输出与输入相同）"
echo "  ⚠️  篇章层未改进翻译（输出与输入相同）"

