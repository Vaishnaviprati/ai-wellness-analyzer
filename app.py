from flask import Flask, render_template, request, jsonify, session
import sqlite3
import os
import re
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# AI STRESS & WELLNESS ANALYZER
# ============================================================
#
# AI COMPONENTS
# 1. Behavioural stress scoring
# 2. AI-generated personalized insights
# 3. Human-in-the-loop conversational wellness assistant
# 4. Session-based conversation memory
# 5. SQLite assessment history
#
# ============================================================

app = Flask(__name__)

# Required for session-based chatbot memory
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ai-wellness-development-secret-key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mental_wellness.db")


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS wellness_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep TEXT NOT NULL,
            workload TEXT NOT NULL,
            mood TEXT NOT NULL,
            social TEXT NOT NULL,
            energy TEXT NOT NULL,
            appetite TEXT NOT NULL,
            stress_score REAL NOT NULL,
            stress_level TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# QUESTION VALUES
# ============================================================

QUESTION_OPTIONS = {

    "sleep": {
        "good": 0,
        "average": 1,
        "poor": 2
    },

    "workload": {
        "light": 0,
        "medium": 1,
        "heavy": 2
    },

    "mood": {
        "happy": 0,
        "neutral": 1,
        "sad": 2
    },

    "social": {
        "often": 0,
        "sometimes": 1,
        "rarely": 2
    },

    "energy": {
        "high": 0,
        "medium": 1,
        "low": 2
    },

    "appetite": {
        "normal": 0,
        "reduced": 1
    }
}


# ============================================================
# STRESS SCORE
# ============================================================

def calculate_score(data):

    sleep = QUESTION_OPTIONS["sleep"][data["sleep"]] / 2 * 100
    workload = QUESTION_OPTIONS["workload"][data["workload"]] / 2 * 100
    mood = QUESTION_OPTIONS["mood"][data["mood"]] / 2 * 100
    social = QUESTION_OPTIONS["social"][data["social"]] / 2 * 100
    energy = QUESTION_OPTIONS["energy"][data["energy"]] / 2 * 100
    appetite = QUESTION_OPTIONS["appetite"][data["appetite"]] * 100

    score = (
        sleep * 0.22 +
        workload * 0.22 +
        mood * 0.20 +
        social * 0.12 +
        energy * 0.16 +
        appetite * 0.08
    )

    score = round(float(score), 1)

    if score < 35:
        level = "Low"
    elif score < 65:
        level = "Moderate"
    else:
        level = "High"

    return score, level


# ============================================================
# FACTOR ANALYSIS
# ============================================================

def get_factors(data):

    return {

        "Sleep": round(
            QUESTION_OPTIONS["sleep"][data["sleep"]] / 2 * 100
        ),

        "Workload": round(
            QUESTION_OPTIONS["workload"][data["workload"]] / 2 * 100
        ),

        "Mood": round(
            QUESTION_OPTIONS["mood"][data["mood"]] / 2 * 100
        ),

        "Social": round(
            QUESTION_OPTIONS["social"][data["social"]] / 2 * 100
        ),

        "Energy": round(
            QUESTION_OPTIONS["energy"][data["energy"]] / 2 * 100
        ),

        "Appetite": round(
            QUESTION_OPTIONS["appetite"][data["appetite"]] * 100
        )
    }


# ============================================================
# AI INSIGHT ENGINE
# ============================================================

def generate_insights(data, score, level, factors):

    insights = []
    actions = []

    sorted_factors = sorted(
        factors.items(),
        key=lambda x: x[1],
        reverse=True
    )

    strongest_name = sorted_factors[0][0]
    strongest_value = sorted_factors[0][1]

    if level == "High":

        insights.append(
            "Your current responses show a higher stress pattern. "
            "Several behavioural indicators are contributing to the result."
        )

    elif level == "Moderate":

        insights.append(
            "Your responses indicate a moderate stress pattern. "
            "A few areas of your daily routine may benefit from attention."
        )

    else:

        insights.append(
            "Your responses indicate a relatively lower stress pattern. "
            "Your current routine appears reasonably balanced."
        )

    insights.append(
        f"{strongest_name} is currently the strongest contributing "
        f"factor in the assessment at approximately {strongest_value}%."
    )

    if data["sleep"] == "poor":

        insights.append(
            "Sleep appears to be an important area for improvement."
        )

        actions.append(
            "Maintain a consistent sleep and wake schedule."
        )

    if data["workload"] == "heavy":

        insights.append(
            "A heavy workload may be increasing your current stress load."
        )

        actions.append(
            "Break large tasks into smaller priorities and schedule short breaks."
        )

    if data["mood"] == "sad":

        insights.append(
            "Your reported mood may be contributing to the current stress pattern."
        )

        actions.append(
            "Make time for activities and people that support your wellbeing."
        )

    if data["social"] == "rarely":

        insights.append(
            "Limited social interaction is appearing as a potential contributing factor."
        )

        actions.append(
            "Consider connecting with a trusted friend, family member or mentor."
        )

    if data["energy"] == "low":

        insights.append(
            "Low energy is contributing to the overall assessment."
        )

        actions.append(
            "Take regular movement breaks and maintain hydration and regular meals."
        )

    if data["appetite"] == "reduced":

        insights.append(
            "Reduced appetite has been identified as one of the behavioural indicators."
        )

        actions.append(
            "Try to maintain regular balanced meals and adequate hydration."
        )

    if not actions:

        actions.append(
            "Continue monitoring sleep, workload, mood, energy and daily recovery."
        )

    if level == "High":

        actions.insert(
            0,
            "Consider reducing immediate workload and speaking with a qualified professional if the pattern persists."
        )

    elif level == "Moderate":

        actions.insert(
            0,
            "Introduce small recovery breaks throughout the day and monitor your stress pattern."
        )

    else:

        actions.insert(
            0,
            "Continue the habits that are supporting your current wellbeing."
        )

    return insights[:5], actions[:6]


# ============================================================
# CHATBOT
# HUMAN-IN-THE-LOOP ENGINE
# ============================================================

CHAT_TOPICS = [

    "workload",
    "sleep",
    "mood",
    "energy",
    "social",
    "anxiety",
    "stress",
    "recovery"
]


def clean_text(text):

    text = (text or "").lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


def detect_topic(text):

    text = clean_text(text)

    topic_words = {

        "workload": [
            "work",
            "workload",
            "deadline",
            "deadlines",
            "college",
            "assignment",
            "assignments",
            "exam",
            "exams",
            "project",
            "projects",
            "job",
            "office",
            "pressure"
        ],

        "sleep": [
            "sleep",
            "sleeping",
            "insomnia",
            "awake",
            "waking",
            "night",
            "rest"
        ],

        "mood": [
            "sad",
            "sadness",
            "low",
            "unhappy",
            "upset",
            "down",
            "cry",
            "crying",
            "lonely",
            "loneliness",
            "mood"
        ],

        "energy": [
            "tired",
            "tiredness",
            "exhausted",
            "exhaustion",
            "drained",
            "energy",
            "fatigue",
            "weak"
        ],

        "social": [
            "friend",
            "friends",
            "family",
            "social",
            "alone",
            "lonely",
            "people",
            "relationship",
            "relationships"
        ],

        "anxiety": [
            "anxiety",
            "anxious",
            "worry",
            "worried",
            "worrying",
            "nervous",
            "panic",
            "restless",
            "overthinking"
        ],

        "stress": [
            "stress",
            "stressed",
            "pressure",
            "overwhelmed",
            "overwhelming"
        ],

        "recovery": [
            "relax",
            "relaxing",
            "calm",
            "better",
            "help",
            "advice",
            "recover",
            "recovery",
            "cope"
        ]
    }

    scores = {}

    for topic, words in topic_words.items():

        count = 0

        for word in words:

            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text
            ):
                count += 1

        scores[topic] = count

    best_topic = max(
        scores,
        key=scores.get
    )

    if scores[best_topic] == 0:

        return "general"

    return best_topic


def detect_emotional_state(text):

    text = clean_text(text)

    positive = [
        "good",
        "better",
        "happy",
        "fine",
        "okay",
        "great",
        "calm",
        "relaxed",
        "positive"
    ]

    negative = [
        "bad",
        "sad",
        "stressed",
        "stress",
        "anxious",
        "worried",
        "overwhelmed",
        "tired",
        "exhausted",
        "lonely",
        "upset",
        "low"
    ]

    positive_count = sum(
        1 for word in positive
        if re.search(r"\b" + re.escape(word) + r"\b", text)
    )

    negative_count = sum(
        1 for word in negative
        if re.search(r"\b" + re.escape(word) + r"\b", text)
    )

    if negative_count > positive_count:

        return "negative"

    if positive_count > negative_count:

        return "positive"

    return "neutral"


def is_greeting(text):

    text = clean_text(text)

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    return text in greetings


def is_thanks(text):

    text = clean_text(text)

    return any(
        phrase in text
        for phrase in [
            "thank you",
            "thanks",
            "thank u",
            "thx"
        ]
    )


def is_yes(text):

    text = clean_text(text)

    return text in [
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "okay",
        "ok",
        "yes please"
    ]


def is_no(text):

    text = clean_text(text)

    return text in [
        "no",
        "nope",
        "not really",
        "nah"
    ]


# ============================================================
# SESSION INITIALIZATION
# ============================================================

def get_chat_state():

    if "chat_state" not in session:

        session["chat_state"] = {

            "started": False,

            "topic": None,

            "stage": "opening",

            "asked_topics": [],

            "answered_topics": [],

            "messages": [],

            "user_profile": {},

            "turn_count": 0
        }

    return session["chat_state"]


def save_chat_state(state):

    session["chat_state"] = state

    session.modified = True


# ============================================================
# CONVERSATION QUESTIONS
# ============================================================

TOPIC_QUESTIONS = {

    "workload": [
        "What is putting the most pressure on you right now — deadlines, too many tasks, college/work expectations, or something else?",
        "How has that workload been affecting you — mainly mentally, physically, or both?",
        "Do you feel that the workload is something you can manage with better planning, or does it currently feel beyond your control?"
    ],

    "sleep": [
        "How has your sleep been recently — are you sleeping less than usual, having trouble falling asleep, or waking up during the night?",
        "About how many hours are you sleeping on a typical night?",
        "Do you feel rested when you wake up, or do you still feel tired?"
    ],

    "mood": [
        "How would you describe your mood recently — mostly low, irritated, emotionally drained, or something else?",
        "Has this change in mood been happening for a few days, or has it been going on for longer?",
        "Is there something specific that you feel is contributing to how you have been feeling?"
    ],

    "energy": [
        "How has your energy been recently — slightly lower than usual or are you feeling exhausted most of the day?",
        "Does the low energy make it difficult to concentrate or complete your normal activities?",
        "How has your sleep been while you have been feeling this tired?"
    ],

    "social": [
        "Have you been getting enough time with people you feel comfortable talking to?",
        "When you feel stressed, do you usually talk to someone or prefer to deal with it on your own?",
        "Is feeling alone contributing to how you have been feeling?"
    ],

    "anxiety": [
        "What tends to trigger the worry most — work or college, relationships, future plans, or something else?",
        "When you start worrying, is it difficult to switch your mind off?",
        "Does the worrying affect your sleep, concentration, or daily activities?"
    ],

    "stress": [
        "What feels like the biggest source of stress for you right now?",
        "How has that stress been affecting your daily routine — sleep, concentration, energy, or mood?",
        "Has this stress been temporary because of something happening right now, or has it been building for a while?"
    ],

    "recovery": [
        "What have you already tried to feel better — taking a break, sleeping, talking to someone, exercising, or something else?",
        "Which part of your routine feels hardest to maintain when you are stressed?",
        "Would you like to focus on managing your workload, improving recovery, or understanding your stress pattern?"
    ]
}


def get_next_question(topic, asked_count):

    questions = TOPIC_QUESTIONS.get(
        topic,
        TOPIC_QUESTIONS["stress"]
    )

    if asked_count < len(questions):

        return questions[asked_count]

    return None


# ============================================================
# HUMAN-IN-THE-LOOP RESPONSE ENGINE
# ============================================================

def generate_chat_response(message, state):

    text = clean_text(message)

    topic = detect_topic(text)

    emotional_state = detect_emotional_state(text)

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if is_greeting(text):

        state["started"] = True
        state["stage"] = "conversation"

        return (
            "Hi! I'm your AI wellness assistant. "
            "I'm here to understand what you're experiencing rather than just give you a generic recommendation. "
            "What's been bothering you the most lately?"
        )


    # --------------------------------------------------------
    # THANK YOU
    # --------------------------------------------------------

    if is_thanks(text):

        return (
            "You're welcome. I'm glad you shared that. "
            "If you want, we can keep exploring what's affecting you or you can use the Assessment section for a structured stress analysis."
        )


    # --------------------------------------------------------
    # FIRST MESSAGE
    # --------------------------------------------------------

    if not state["started"]:

        state["started"] = True
        state["stage"] = "conversation"

        if topic != "general":

            state["topic"] = topic

            if topic not in state["asked_topics"]:

                state["asked_topics"].append(topic)

            question = get_next_question(
                topic,
                0
            )

            state["turn_count"] += 1

            return (
                f"I understand. It sounds like {topic} may be an important part of what you're experiencing. "
                f"{question}"
            )

        state["topic"] = None

        state["turn_count"] += 1

        return (
            "I want to understand this properly rather than assume what you're going through. "
            "Can you tell me a little more about what has been making you feel this way?"
        )


    # --------------------------------------------------------
    # GENERAL MESSAGE AFTER CONVERSATION STARTED
    # --------------------------------------------------------

    if topic == "general":

        current_topic = state.get("topic")

        if current_topic:

            answered = state.get(
                "answered_topics",
                []
            )

            if current_topic not in answered:

                answered.append(current_topic)

                state["answered_topics"] = answered

            question = get_next_question(
                current_topic,
                len(answered)
            )

            if question:

                state["turn_count"] += 1

                return (
                    "That helps me understand the situation better. "
                    + question
                )

        state["turn_count"] += 1

        return (
            "I’m listening. Could you tell me a little more about what has been affecting you recently?"
        )


    # --------------------------------------------------------
    # NEW TOPIC DETECTED
    # --------------------------------------------------------

    if topic != state.get("topic"):

        previous_topic = state.get("topic")

        state["topic"] = topic

        if topic not in state["asked_topics"]:

            state["asked_topics"].append(topic)

        question = get_next_question(
            topic,
            0
        )

        state["turn_count"] += 1

        if previous_topic:

            return (
                f"I understand. It sounds like {topic} is also important here. "
                f"Let's look at that as well. {question}"
            )

        return (
            f"Thanks for explaining that. Let's look at the {topic} part first. "
            f"{question}"
        )


    # --------------------------------------------------------
    # SAME TOPIC — CONTINUE CONVERSATION
    # --------------------------------------------------------

    current_topic = state.get("topic")

    if current_topic:

        answered_topics = state.get(
            "answered_topics",
            []
        )

        if current_topic not in answered_topics:

            answered_topics.append(current_topic)

        state["answered_topics"] = answered_topics

        # Number of questions already asked for this topic
        topic_question_count = sum(
            1
            for msg in state.get("messages", [])
            if msg.get("role") == "assistant"
            and msg.get("topic") == current_topic
            and msg.get("is_question") is True
        )

        next_question = get_next_question(
            current_topic,
            topic_question_count
        )

        state["turn_count"] += 1

        if next_question:

            if emotional_state == "negative":

                return (
                    "That sounds difficult to deal with. "
                    + next_question
                )

            return (
                "Thanks for telling me that. "
                + next_question
            )

    # --------------------------------------------------------
    # CONVERSATION COMPLETE / SUMMARY
    # --------------------------------------------------------

    state["stage"] = "summary"

    state["turn_count"] += 1

    topic = state.get("topic")

    if topic == "workload":

        return (
            "I have a clearer picture now. Your workload appears to be an important part of the stress pattern you're describing. "
            "Rather than guessing the severity, I recommend running the AI Stress Assessment as well. "
            "That will compare workload with sleep, mood, energy, social interaction and appetite."
        )

    if topic == "sleep":

        return (
            "I have a better understanding of your sleep concern now. "
            "Because sleep can interact with energy, mood and stress, the AI assessment can help you see the broader pattern."
        )

    if topic == "anxiety":

        return (
            "I understand the worry you're describing. "
            "The next useful step would be to look at how it connects with your sleep, workload, mood and energy through the assessment."
        )

    if topic == "mood":

        return (
            "Thank you for explaining that. "
            "Your mood is one part of the overall wellbeing pattern, so the assessment can help check whether other factors are contributing as well."
        )

    return (
        "Thanks for talking this through with me. "
        "I have a better understanding of what you're experiencing. "
        "If you'd like a structured view, run the AI Stress Assessment and we can use those results together."
    )


# ============================================================
# CHAT ROUTE
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    message = str(
        data.get("message", "")
    ).strip()

    if not message:

        return jsonify({
            "error": "Please enter a message."
        }), 400

    state = get_chat_state()

    # Store user message
    state["messages"].append({

        "role": "user",

        "message": message,

        "timestamp":
            datetime.now().strftime(
                "%H:%M:%S"
            )
    })

    reply = generate_chat_response(
        message,
        state
    )

    # Determine whether reply contains a follow-up question
    is_question = "?" in reply

    state["messages"].append({

        "role": "assistant",

        "message": reply,

        "topic": state.get("topic"),

        "is_question": is_question,

        "timestamp":
            datetime.now().strftime(
                "%H:%M:%S"
            )
    })

    save_chat_state(state)

    return jsonify({

        "reply": reply,

        "topic":
            state.get("topic"),

        "stage":
            state.get("stage"),

        "turn_count":
            state.get("turn_count", 0),

        "conversation_active":
            True

    })


# ============================================================
# RESET CHAT
# ============================================================

@app.route("/chat/reset", methods=["POST"])
def reset_chat():

    session.pop(
        "chat_state",
        None
    )

    return jsonify({
        "success": True
    })


# ============================================================
# CHAT HISTORY FOR CURRENT SESSION
# ============================================================

@app.route("/chat/history")
def chat_history():

    state = get_chat_state()

    return jsonify(
        state.get(
            "messages",
            []
        )
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(
        silent=True
    ) or {}

    required = [
        "sleep",
        "workload",
        "mood",
        "social",
        "energy",
        "appetite"
    ]

    if not all(
        key in data
        for key in required
    ):

        return jsonify({
            "error":
                "Please answer all assessment questions."
        }), 400

    try:

        score, level = calculate_score(
            data
        )

        factors = get_factors(
            data
        )

        insights, actions = generate_insights(
            data,
            score,
            level,
            factors
        )

        conn = get_connection()

        cursor = conn.execute("""

            INSERT INTO wellness_data (

                sleep,
                workload,
                mood,
                social,
                energy,
                appetite,
                stress_score,
                stress_level

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)

        """, (

            data["sleep"],
            data["workload"],
            data["mood"],
            data["social"],
            data["energy"],
            data["appetite"],
            score,
            level

        ))

        assessment_id = cursor.lastrowid

        conn.commit()

        conn.close()

        return jsonify({

            "id":
                assessment_id,

            "score":
                score,

            "level":
                level,

            "factors":
                factors,

            "insights":
                insights,

            "actions":
                actions

        })

    except Exception as exc:

        return jsonify({
            "error":
                str(exc)
        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = get_connection()

    rows = conn.execute("""

        SELECT

            id,
            stress_score,
            stress_level,
            sleep,
            workload,
            mood,
            social,
            energy,
            appetite,
            created_at

        FROM wellness_data

        ORDER BY id DESC

        LIMIT 20

    """).fetchall()

    conn.close()

    return jsonify([

        dict(row)

        for row in rows

    ])


# ============================================================
# SUMMARY
# ============================================================

@app.route("/summary")
def summary():

    conn = get_connection()

    row = conn.execute("""

        SELECT

            COUNT(*) AS total,

            AVG(stress_score)
                AS average_score,

            MAX(stress_score)
                AS highest_score,

            MIN(stress_score)
                AS lowest_score

        FROM wellness_data

    """).fetchone()

    latest = conn.execute("""

        SELECT

            stress_score,
            stress_level

        FROM wellness_data

        ORDER BY id DESC

        LIMIT 1

    """).fetchone()

    conn.close()

    return jsonify({

        "total":
            row["total"] or 0,

        "average_score":
            round(
                row["average_score"] or 0,
                1
            ),

        "highest_score":
            round(
                row["highest_score"] or 0,
                1
            ),

        "lowest_score":
            round(
                row["lowest_score"] or 0,
                1
            ),

        "latest_score":
            latest["stress_score"]
            if latest
            else None,

        "latest_level":
            latest["stress_level"]
            if latest
            else None

    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    init_db()

    print("=" * 60)

    print(
        "AI STRESS & WELLNESS ANALYZER"
    )

    print(
        "AI ENGINE: BEHAVIOURAL SCORING"
    )

    print(
        "CONVERSATION ENGINE: HUMAN-IN-THE-LOOP"
    )

    print(
        "MEMORY: SESSION BASED"
    )

    print(
        "DATABASE: SQLITE"
    )

    print("=" * 60)

    app.run(
        debug=True
    )