import pandas as pd
import random

# Define moods and activities
mood_activities = {
    "Sadness": [
        "Listen to calming music", "Write down your feelings", "Watch a comforting movie",
        "Take a walk in nature", "Call a friend or family member", "Try meditation", "Engage in a creative hobby"
    ],
    "Happiness": [
        "Celebrate with friends", "Write about your joyful moment", "Go for a fun outing",
        "Share your happiness on social media", "Practice gratitude", "Listen to upbeat music", "Dance freely"
    ],
    "Anger": [
        "Try deep breathing exercises", "Go for a run or exercise", "Write about your frustration",
        "Listen to relaxing music", "Practice mindfulness", "Vent to a close friend", "Engage in a calming activity"
    ],
    "Neutral": [
        "Explore a new hobby", "Read an interesting book", "Try learning something new",
        "Go for a casual walk", "Watch an inspiring documentary", "Organize your workspace", "Plan your week ahead"
    ],
    "Love": [
        "Write a heartfelt message to someone", "Spend quality time with loved ones", "Express gratitude",
        "Do something kind for someone", "Reminisce on happy memories", "Watch a romantic movie", "Send a surprise gift"
    ],
    "Joy": [
        "Capture this moment in a journal", "Express gratitude", "Enjoy your favorite treat",
        "Plan a small celebration", "Sing your favorite song", "Share your happiness with others", "Engage in a fun activity"
    ],
    "Fear": [
        "Talk to someone about your fear", "Practice deep breathing", "Write down your thoughts",
        "Engage in positive affirmations", "Distract yourself with a light activity", "Listen to calming music",
        "Watch an uplifting movie"
    ],
    "Surprise": [
        "Embrace the unexpected moment", "Write about how you feel", "Share the surprise with a friend",
        "Take a deep breath and process it", "Celebrate a good surprise", "Learn from an unexpected situation",
        "Reflect on how surprises impact you"
    ]
}

# Generating 500+ synthetic diary entries
diary_entries = []
for mood, activities in mood_activities.items():
    for _ in range(70):  # Generating approx. 70 diary entries per mood
        diary_entry = f"{random.choice(['I', 'Today, I', 'Lately, I'])} {random.choice(['feel', 'am feeling', 'have been feeling'])} {mood.lower()} {random.choice(['and unsure what to do.', 'and looking for something to lift my mood.', 'but I hope it gets better.'])}"
        suggested_activity = random.choice(activities)
        diary_entries.append([diary_entry, mood, suggested_activity])

# Convert to DataFrame
df = pd.DataFrame(diary_entries, columns=["Diary Entry", "Detected Mood", "Suggested Activity"])

# Save to CSV
df.to_csv("recommendations.csv", index=False)
print("Dataset generated and saved as recommendations.csv")
