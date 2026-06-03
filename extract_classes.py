import os
import json

# 1. مسار مجلد التدريب
train_dir = 'split_dataset/train'

# 2. الحصول على أسماء الفئات وترتيبها
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print(f"عدد الفئات: {len(classes)}")
print("الأسماء: \n", classes)

# 3. حفظ الأسماء في ملف classes.json
with open('classes.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)
print("\nتم حفظ أسماء الفئات في ملف classes.json")