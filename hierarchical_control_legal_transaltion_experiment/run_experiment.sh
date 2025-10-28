export OPENAI_EMBED_MODEL=doubao-embedding-text-240715

# OpenAI API 配置
export OPENAI_API_KEY=258be12b-4726-448f-976b-59491ef007f2
export OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export OPENAI_API_MODEL=deepseek-v3-1-terminus


#python run_experiment.py --test-set dataset/processed/test_set_zh_en_sample100.json --max-concurrent 40 --ablations baseline --output-dir outputs/translation-baseline
#python run_experiment.py --test-set dataset/processed/test_set_zh_en_sample100.json --max-concurrent 40 --ablations terminology  --output-dir outputs/translation-terminology
#python run_experiment.py --test-set dataset/processed/test_set_zh_en_sample100.json --max-concurrent 40 --ablations terminology_syntax --output-dir outputs/translation-terminology_syntax
python run_experiment.py --test-set dataset/processed/test_set_zh_en_sample20.json --max-concurrent 40 --ablations full --output-dir outputs/translation-full-sample-20



