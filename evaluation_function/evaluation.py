import json
import logging
import os
import sys
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv
from lf_toolkit.evaluation import Result, Params

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


def process_prompt(prompt, question, answer):
    prompt = prompt.replace("{{answer}}", str(answer))
    prompt = prompt.replace("{{question}}", str(question) or "")
    prompt = prompt.strip()
    if prompt and not prompt.endswith('.'):
        prompt += '.'
    return prompt


def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """

    logger.debug("evaluation_function called: response=%r, answer=%r", response, answer)

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )

    question = params.get("question")
    logger.debug("question=%r, model=%r", question, params.get("model"))

    main_prompt = process_prompt(params['main_prompt'], question, answer)
    default_prompt = process_prompt(params['default_prompt'], question, answer)
    feedback_prompt = process_prompt(params['feedback_prompt'], question, answer)

    if params['feedback_prompt'].strip():
        logger.debug("running combined correctness + feedback check")
        combined_system = (
            f"{main_prompt} {default_prompt} {feedback_prompt} "
            'Output your response as a JSON object with exactly two fields: '
            '"is_correct" (boolean, true if the student response is correct, false otherwise) '
            'and "feedback" (string, feedback for the student).'
        )
        combined_result = client.chat.completions.create(
            model=params['model'],
            messages=[
                {"role": "system", "content": combined_system},
                {"role": "user", "content": response},
            ],
        )
        raw = combined_result.choices[0].message.content.strip()
        logger.debug("combined result raw: %r", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("failed to parse combined result as JSON: %r", raw)
            raise ValueError(f"Model did not return valid JSON: {e}") from e
        is_correct = bool(data["is_correct"])
        feedback_text = str(data.get("feedback", ""))
        logger.debug("is_correct=%s, feedback=%r", is_correct, feedback_text)
        result = Result(is_correct=is_correct)
        result.add_feedback("feedback", feedback_text)
        return result
    else:
        logger.debug("running correctness check")
        correctness_result = client.chat.completions.create(
            model=params['model'],
            messages=[
                {"role": "system", "content": main_prompt + " " + default_prompt},
                {"role": "user", "content": response},
            ],
        )
        correctness_verdict = correctness_result.choices[0].message.content.strip()
        is_correct = correctness_verdict.lower() == "true"
        logger.debug("correctness verdict: %r -> is_correct=%s", correctness_verdict, is_correct)
        return Result(is_correct=is_correct)
