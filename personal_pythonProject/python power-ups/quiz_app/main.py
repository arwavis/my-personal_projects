import json

import questions


def load_questions(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def ask_question(question):
    print(question['question'])
    if question['type'] == 'multiple_choice' or question['type'] == 'multiple_response':
        for idx, option in enumerate(question['options']):
            print(f"{idx + 1}. {option}")

    user_answer = input("Your answer: ")
    return user_answer


def check_answer(question, user_answer):
    if question['type'] == 'multiple_choice':
        correct_answer = question['options'].index(question['answer']) + 1
        return str(correct_answer) == user_answer.strip()
    elif question['type'] == 'true_false':
        return question['answer'].lower() == user_answer.lower()
    elif question['type'] == 'multiple_response':
        correct_answers = [str(question['options'].index(ans) + 1) for ans in question['answers']]
        user_answers = user_answer.split()
        return sorted(correct_answers) == sorted(user_answers)


def run_quiz(filename):
    questions = load_questions(filename)
    score = 0

    for question in questions:
        user_answer = ask_question(question)
        if check_answer(question, user_answer):
            print("Correct!")
            score += 1
        else:
            print("Wrong!")

    print(f"Your final score is {score}/{len(questions)}")


# You can start the quiz by calling run_quiz('questions.json')
run_quiz('questions.json')
