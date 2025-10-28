#!/bin/bash
# 质量评估功能使用示例

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================"
echo "质量评估功能使用示例"
echo -e "======================================================================${NC}"

# 示例1：简单评估
echo -e "\n${GREEN}示例1：基本评估${NC}"
echo "命令："
echo 'python run_translation.py \'
echo '  --source "劳动者享有平等就业的权利。" \'
echo '  --reference "Workers shall have the right to equal employment." \'
echo '  --evaluate'
echo ""

python run_translation.py \
  --source "劳动者享有平等就业的权利。" \
  --reference "Workers shall have the right to equal employment." \
  --evaluate

# 示例2：完整系统评估
echo -e "\n\n${GREEN}示例2：评估完整系统（术语+句法+篇章）${NC}"
echo "命令："
echo 'python run_translation.py \'
echo '  --source "用人单位招用劳动者，不得扣押劳动者的居民身份证。" \'
echo '  --reference "When recruiting workers, an employing unit must not seize the workers resident identity cards." \'
echo '  --hierarchical \'
echo '  --use-termbase \'
echo '  --evaluate \'
echo '  --verbose'
echo ""

python run_translation.py \
  --source "用人单位招用劳动者，不得扣押劳动者的居民身份证。" \
  --reference "When recruiting workers, an employing unit must not seize the workers resident identity cards." \
  --hierarchical \
  --use-termbase \
  --evaluate \
  --verbose

echo -e "\n${BLUE}======================================================================"
echo "示例完成！"
echo -e "======================================================================${NC}"
echo ""
echo "提示："
echo "  - 使用 --evaluate 启用质量评估"
echo "  - 使用 --reference 提供参考译文"
echo "  - 使用 --verbose 查看详细分析"
echo "  - 使用 --output 保存完整结果到JSON文件"
echo ""

