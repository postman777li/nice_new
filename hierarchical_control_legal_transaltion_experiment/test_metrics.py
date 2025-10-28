#!/usr/bin/env python3
"""
测试现代机器翻译评估指标
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_basic_metrics():
    """测试基础指标（BLEU, chrF++）"""
    print("="*60)
    print("测试基础指标: BLEU 和 chrF++")
    print("="*60)
    
    from src.metrics import BLEUMetric, ChrFMetric
    
    # 测试数据
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    reference = "Contracting parties must comply with all provisions of this agreement."
    
    print(f"\n源文本: {source}")
    print(f"预测翻译: {prediction}")
    print(f"参考翻译: {reference}\n")
    
    # BLEU
    print("计算 BLEU...")
    bleu = BLEUMetric(tokenize='intl')
    bleu_score = bleu.sentence_score(prediction, reference)
    print(f"✓ BLEU: {bleu_score:.2f}")
    
    # chrF++
    print("\n计算 chrF++...")
    chrf = ChrFMetric()
    chrf_score = chrf.sentence_score(prediction, reference)
    print(f"✓ chrF++: {chrf_score:.2f}")
    
    print("\n" + "="*60)
    print("✅ 基础指标测试完成")
    print("="*60)


def test_metric_suite():
    """测试指标套件"""
    print("\n" + "="*60)
    print("测试指标套件（快速指标）")
    print("="*60)
    
    from src.metrics import MetricSuite
    
    # 创建快速指标套件（只用BLEU和chrF，不需要GPU）
    suite = MetricSuite(metrics=['bleu', 'chrf'])
    
    # 测试数据
    source = "劳动者享有平等就业的权利。"
    prediction = "Workers have the right to equal employment."
    reference = "Laborers are entitled to the right to equal employment."
    
    print(f"\n源文本: {source}")
    print(f"预测翻译: {prediction}")
    print(f"参考翻译: {reference}\n")
    
    print("计算所有指标...")
    scores = suite.compute(source, prediction, reference)
    
    print("\n评估结果:")
    for metric, score in scores.items():
        print(f"  {metric:15s}: {score:6.2f}")
    
    print("\n" + "="*60)
    print("✅ 指标套件测试完成")
    print("="*60)


def test_advanced_metrics():
    """测试高级指标（需要模型下载）"""
    print("\n" + "="*60)
    print("测试高级指标（需要下载模型）")
    print("="*60)
    print("\n⚠️  注意：首次运行会下载模型，可能需要较长时间")
    
    response = input("\n是否继续测试高级指标 (BERTScore, COMET)? [y/N]: ")
    if response.lower() != 'y':
        print("跳过高级指标测试")
        return
    
    from src.metrics import BERTScoreMetric, COMETMetric
    
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    reference = "Contracting parties must comply with all provisions of this agreement."
    
    # BERTScore
    try:
        print("\n计算 BERTScore...")
        bertscore = BERTScoreMetric(model_type="xlm-roberta-base")  # 使用较小的模型
        scores = bertscore.compute([prediction], [reference])
        print(f"✓ BERTScore F1: {scores['f1']:.4f}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall: {scores['recall']:.4f}")
    except Exception as e:
        print(f"✗ BERTScore 失败: {e}")
    
    # COMET
    try:
        print("\n计算 COMET (这可能需要几分钟)...")
        comet = COMETMetric(model_name="Unbabel/wmt22-comet-da", gpus=0)
        comet_score = comet.sentence_score(source, prediction, reference)
        print(f"✓ COMET: {comet_score:.4f}")
    except Exception as e:
        print(f"✗ COMET 失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 高级指标测试完成")
    print("="*60)


def test_gemba_metrics():
    """测试GEMBA指标（需要API密钥）"""
    print("\n" + "="*60)
    print("测试GEMBA指标（需要OpenAI API）")
    print("="*60)
    print("\n⚠️  注意：GEMBA使用GPT-4 API，会产生费用")
    
    response = input("\n是否继续测试GEMBA指标 (GEMBA-MQM, GEMBA-DA)? [y/N]: ")
    if response.lower() != 'y':
        print("跳过GEMBA指标测试")
        return
    
    from src.metrics import GEMBAMetric
    
    source = "合同双方应当遵守本协议的所有条款。"
    prediction = "The parties shall comply with all terms of this agreement."
    
    # GEMBA-DA (推荐用于快速评估)
    try:
        print("\n1. 计算 GEMBA-DA (直接评估)...")
        da_metric = GEMBAMetric(method="GEMBA-DA", model="gpt-4")
        da_score = da_metric.sentence_score(source, prediction, "Chinese", "English")
        print(f"✓ GEMBA-DA 分数: {da_score:.2f}/100")
    except Exception as e:
        print(f"✗ GEMBA-DA 失败: {e}")
    
    # GEMBA-MQM (详细错误分析)
    try:
        print("\n2. 计算 GEMBA-MQM (详细错误分析)...")
        mqm_metric = GEMBAMetric(method="GEMBA-MQM", model="gpt-4")
        mqm_result = mqm_metric.compute([source], [prediction], "Chinese", "English")
        print(f"✓ GEMBA-MQM 分数: {mqm_result['mean']:.2f}/100")
        if mqm_result['results']:
            result = mqm_result['results'][0]
            print(f"  错误数量: Minor={result.get('error_count', {}).get('minor', 0)}, "
                  f"Major={result.get('error_count', {}).get('major', 0)}, "
                  f"Critical={result.get('error_count', {}).get('critical', 0)}")
    except Exception as e:
        print(f"✗ GEMBA-MQM 失败: {e}")
    
    print("\n" + "="*60)
    print("✅ GEMBA指标测试完成")
    print("="*60)


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print(" "*15 + "机器翻译评估指标测试")
    print("="*70)
    
    try:
        # 1. 测试基础指标
        test_basic_metrics()
        
        # 2. 测试指标套件
        test_metric_suite()
        
        # 3. 测试高级指标（可选）
        test_advanced_metrics()
        
        # 4. 测试GEMBA指标（可选）
        test_gemba_metrics()
        
        print("\n" + "="*70)
        print(" "*20 + "所有测试完成！")
        print("="*70)
        print("\n💡 提示:")
        print("  1. 基础指标 (BLEU, chrF++) 已可用，速度快")
        print("  2. 高级指标 (BERTScore, COMET) 需要下载模型")
        print("  3. GEMBA指标 (MQM, DA) 使用GPT-4 API，最接近人工评估")
        print("  4. 推荐组合: BLEU + chrF + COMET (平衡速度和质量)")
        print("  5. 在实验中使用: 修改 metrics.py 整合这些指标")
        print("  6. 查看文档: src/metrics/README.md")
        print()
        
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

