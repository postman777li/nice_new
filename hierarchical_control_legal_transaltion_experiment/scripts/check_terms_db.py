#!/usr/bin/env python3
"""
检查术语数据库内容
"""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else 'terms_zh_en.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 总数
cursor.execute('SELECT COUNT(*) FROM terms')
total = cursor.fetchone()[0]
print(f'✅ 术语总数: {total}')

if total > 0:
    # 语言对统计
    cursor.execute('''
        SELECT source_lang, target_lang, COUNT(*) 
        FROM terms 
        GROUP BY source_lang, target_lang
    ''')
    print('\n语言对分布:')
    for row in cursor.fetchall():
        print(f'  {row[0]} -> {row[1]}: {row[2]} 个术语')
    
    # 前5个术语
    cursor.execute('''
        SELECT source_term, target_term, confidence, quality_score, combined_score, law
        FROM terms 
        LIMIT 5
    ''')
    print('\n前5个术语:')
    for row in cursor.fetchall():
        print(f'  {row[0]} -> {row[1]}')
        print(f'    置信度:{row[2]:.2f}, 质量:{row[3]:.2f}, 综合:{row[4]:.2f}, 法律:{row[5]}')

conn.close()

