import os
import json

train_dir = 'split_dataset/train'
class_order = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print(f"عدد الفئات: {len(class_order)}")
for i, name in enumerate(class_order):
    print(f"{i}: {name}")

# حفظ الترتيب في ملف JSON
with open('class_order.json', 'w', encoding='utf-8') as f:
    json.dump(class_order, f, ensure_ascii=False, indent=2)

print("\n✅ تم حفظ الترتيب في class_order.json")