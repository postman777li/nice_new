#!/usr/bin/env python3
"""
测试实验脚本 - 通过后端HTTP API发起与轮询实验
"""
import json
import time
import os
import argparse
import requests


def post_json(url: str, payload: dict) -> dict:
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_json(url: str) -> dict:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="通过HTTP接口测试法律翻译实验")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"), help="后端基础URL")
    parser.add_argument("--src", default="合同当事人应当按照约定履行义务。", help="源文本")
    parser.add_argument("--src-lang", default="zh", help="源语言代码")
    parser.add_argument("--tgt-lang", default="en", help="目标语言代码")
    parser.add_argument("--interval", type=float, default=2.0, help="轮询间隔秒数")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print("🚀 通过HTTP API发起实验...")
    payload = {
        "source": args.src,
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "options": {
            "hierarchical": True,
            "useTermBase": True,
            "useRuleTable": True,
            "useTM": True,
            "topK": 2
        }
    }

    # 1) 创建实验
    create_url = f"{base}/api/experiments"
    create_resp = post_json(create_url, payload)
    job_id = create_resp.get("job_id")
    if not job_id:
        print("❌ 未获取到 job_id:", create_resp)
        return
    print(f"🆔 实验ID: {job_id}")

    # 2) 轮询状态
    status_url = f"{base}/api/experiments/{job_id}/status"
    result_url = f"{base}/api/experiments/{job_id}/result"

    while True:
        try:
            status = get_json(status_url)
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
            time.sleep(args.interval)
            continue

        print(f"📊 状态: {status.get('status')} | 进度: {status.get('progress')}% | 阶段: {status.get('current_stage')} | 消息: {status.get('message')}")
        if status.get("status") in ["completed", "failed", "cancelled"]:
            break
        time.sleep(args.interval)

    # 3) 获取结果
    try:
        result = get_json(result_url)
    except requests.HTTPError as e:
        # 可能实验尚未完全落盘，稍后再试一次
        print(f"⚠️ 获取结果失败，重试一次: {e}")
        time.sleep(1.0)
        result = get_json(result_url)

    print("\n" + "=" * 80)
    print("🎉 实验完成!")
    print(f"✅ 成功: {result.get('success')}")
    print(f"📄 最终输出: {result.get('final')}")
    if "duration" in result and result["duration"] is not None:
        print(f"⏱️ 耗时: {result['duration']:.2f}秒")

    print("\n📋 详细追踪:")
    trace = result.get("trace", {})
    for round_name, round_result in trace.items():
        print(f"\n🔸 {round_name.upper()}:")
        if isinstance(round_result, dict):
            for key, value in round_result.items():
                if key != "output":
                    # 避免打印超长内容
                    preview = json.dumps(value, ensure_ascii=False)[:500]
                    print(f"  {key}: {preview}")
        else:
            print(f"  {round_result}")


if __name__ == "__main__":
    main()
