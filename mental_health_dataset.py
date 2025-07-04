# Mental Health Dataset
import random

def create_variations(base_text, variations):
    """Create variations of a base text by replacing key words."""
    results = [base_text]
    for _ in range(variations - 1):
        text = base_text
        if "anxious" in text:
            text = text.replace("anxious", random.choice(["nervous", "worried", "uneasy", "apprehensive"]))
        if "stressed" in text:
            text = text.replace("stressed", random.choice(["overwhelmed", "pressured", "tense", "strained"]))
        if "depressed" in text:
            text = text.replace("depressed", random.choice(["down", "sad", "miserable", "hopeless"]))
        results.append(text)
    return results

# Base statements
anxiety_base = [
    "I feel anxious and worried about my future",
    "I can't stop thinking about my problems",
    "I'm having trouble sleeping due to anxiety",
    "I'm experiencing panic attacks frequently",
    "My heart races when I think about upcoming events",
    "I feel restless and can't sit still",
    "I worry about things that might never happen",
    "I feel tense and on edge most of the time",
    "I avoid situations that make me anxious",
    "I have difficulty concentrating due to worry"
]

depression_base = [
    "I'm feeling depressed and don't want to get out of bed",
    "I feel hopeless about the future",
    "I've lost interest in activities I used to enjoy",
    "I feel worthless and guilty all the time",
    "I have thoughts about harming myself",
    "I feel tired and have no energy",
    "I have difficulty making simple decisions",
    "I feel sad and empty most of the time",
    "I have trouble sleeping or sleeping too much",
    "I've lost or gained significant weight recently"
]

stress_base = [
    "I'm under a lot of stress at work",
    "I feel overwhelmed by my responsibilities",
    "I have trouble relaxing after work",
    "I feel irritable and easily frustrated",
    "I have frequent headaches from stress",
    "I feel like I can't cope with daily tasks",
    "I have trouble managing my time effectively",
    "I feel pressured to meet expectations",
    "I'm stressed about my financial situation",
    "I feel like I'm always in a rush"
]

normal_base = [
    "I feel happy and content with my life",
    "I feel optimistic about the future",
    "I feel calm and peaceful most days",
    "I have good relationships with others",
    "I feel confident in my abilities",
    "I enjoy my daily activities",
    "I have a good work-life balance",
    "I feel grateful for what I have",
    "I have healthy coping mechanisms",
    "I feel well-rested and energetic"
]

# Create variations
texts = []
labels = []

# Add variations of each category
for base_list, label, variations in [
    (anxiety_base, "Anxiety", 5),
    (depression_base, "Depression", 5),
    (stress_base, "Stress", 5),
    (normal_base, "Normal", 5)
]:
    for base_text in base_list:
        variations_list = create_variations(base_text, variations)
        texts.extend(variations_list)
        labels.extend([label] * len(variations_list))

# Add mixed statements with clear primary emotions
mixed_statements = [
    ("I sometimes feel anxious but can manage it", "Anxiety"),
    ("I have good days and bad days with my mood", "Normal"),
    ("I feel stressed but know it's temporary", "Stress"),
    ("I occasionally feel down but bounce back", "Normal"),
    ("I have some worries but they don't control me", "Anxiety"),
    ("I feel pressure but can handle it", "Stress"),
    ("I have moments of anxiety but they pass", "Anxiety"),
    ("I feel stressed but maintain perspective", "Stress"),
    ("I have occasional low moods but stay positive", "Normal"),
    ("I feel overwhelmed sometimes but cope well", "Normal")
]

# Add variations of mixed statements
for text, label in mixed_statements:
    variations_list = create_variations(text, 3)
    texts.extend(variations_list)
    labels.extend([label] * len(variations_list))

# Save the dataset
import pickle

with open('mental_health_dataset.pickle', 'wb') as handle:
    pickle.dump({'texts': texts, 'labels': labels}, handle)

print(f"Dataset saved successfully! Total examples: {len(texts)}")
print("\nClass distribution:")
from collections import Counter
class_dist = Counter(labels)
for label, count in class_dist.items():
    print(f"{label}: {count} examples") 