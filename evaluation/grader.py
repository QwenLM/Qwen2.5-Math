"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py
"""

import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy

import unittest

# from .parser import choice_answer_clean, strip_string
# from parser import choice_answer_clean


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def is_numerical_equal(prediction, reference, include_percentage, is_close):
    if is_digit(prediction) and is_digit(reference):
        prediction = parse_digits(prediction)
        reference = parse_digits(reference)
        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]
        for item in gt_result:
            try:
                if is_close:
                    if numeric_equal(prediction, item):
                        return True
                else:
                    if item == prediction:
                        return True
            except Exception:
                continue
    return False


def process_matrix(prediction, reference):
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)
    return prediction, reference


def has_format(s, start_char, end_char):
    return s.startswith(start_char) and s.endswith(end_char)


def remove_brackets(s, brackets):
    for bracket in brackets:
        s = s.replace(bracket, "")
    return s


def is_list_format(s):
    return regex.match(r"(\(|\[).+(\)|\])", s) is not None


def is_matrix_format(s):
    return (
        (s.startswith("\\begin{pmatrix}") or s.startswith("\\begin{bmatrix}"))
        and (s.endswith("\\end{pmatrix}") or s.endswith("\\end{bmatrix}"))
    )


def compare_lists(prediction, reference, include_percentage, is_close):
    pred_parts = prediction[1:-1].split(",")
    ref_parts = reference[1:-1].split(",")
    if len(pred_parts) == len(ref_parts):
        return all(
            math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
            for i in range(len(pred_parts))
        )
    return False


def compare_matrices(prediction, reference, include_percentage, is_close):
    pred_lines = [
        line.strip()
        for line in prediction[
            len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
        ].split("\\\\")
        if line.strip()
    ]
    ref_lines = [
        line.strip()
        for line in reference[
            len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
        ].split("\\\\")
        if line.strip()
    ]
    if len(pred_lines) == len(ref_lines):
        for pred_line, ref_line in zip(pred_lines, ref_lines):
            pred_parts = pred_line.split("&")
            ref_parts = ref_line.split("&")
            if len(pred_parts) != len(ref_parts):
                return False
            if not all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return False
        return True
    return False


def compare_structures(prediction, reference, include_percentage, is_close):
    if is_list_format(prediction) and is_list_format(reference):
        return compare_lists(prediction, reference, include_percentage, is_close)

    if is_matrix_format(prediction) and is_matrix_format(reference):
        return compare_matrices(prediction, reference, include_percentage, is_close)

    return False


def format_equation(equation: str) -> str:
    parts = equation.split("=")
    return f"{parts[0].strip()} - ({parts[1].strip()})"


def check_symbolic_equality(prediction: str, reference: str) -> bool:
    pred = format_equation(prediction)
    ref = format_equation(reference)
    return symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref)


def check_math_equality(
    prediction: str, reference: str, include_percentage: bool, is_close: bool
) -> bool:
    return math_equal(prediction, reference, include_percentage, is_close)


def is_simple_equation(equation: str) -> bool:
    return equation.count("=") == 1 and len(equation.split("=")[0].strip()) <= 2


def compare_equations(
    prediction: str, reference: str, include_percentage: bool, is_close: bool
) -> bool:
    if prediction.count("=") == 1 and reference.count("=") == 1:
        return check_symbolic_equality(prediction, reference)
    elif is_simple_equation(prediction) and "=" not in reference:
        return check_math_equality(
            prediction.split("=")[1], reference, include_percentage, is_close
        )
    elif is_simple_equation(reference) and "=" not in prediction:
        return check_math_equality(
            prediction, reference.split("=")[1], include_percentage, is_close
        )
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    if is_numerical_equal(prediction, reference, include_percentage, is_close):
        return True

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## pmatrix (amps)
    prediction, reference = process_matrix(prediction, reference)

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (has_format(pred_str, "[", "]") and not has_format(ref_str, "(", ")")) or (
        has_format(pred_str, "(", ")") and not has_format(ref_str, "[", "]")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    brackets_to_remove = ["{", "}", "(", ")"]
    pred_str = remove_brackets(pred_str, brackets_to_remove)
    ref_str = remove_brackets(ref_str, brackets_to_remove)

    if pred_str.lower() == ref_str.lower():
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if is_list_format(prediction) and is_list_format(reference):
        return compare_lists(prediction, reference, include_percentage, is_close)

    if is_matrix_format(prediction) and is_matrix_format(reference):
        return compare_matrices(prediction, reference, include_percentage, is_close)

    if prediction.count("=") == 1 and reference.count("=") == 1:
        return check_symbolic_equality(prediction, reference)
    elif is_simple_equation(prediction) and "=" not in reference:
        return check_math_equality(
            prediction.split("=")[1], reference, include_percentage, is_close
        )
    elif is_simple_equation(reference) and "=" not in prediction:
        return check_math_equality(
            prediction, reference.split("=")[1], include_percentage, is_close
        )

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    # Note that relative tolerance has significant impact
    # on the result of the synthesized GSM-Hard dataset
    # if reference.is_integer():
    #     return isclose(reference, round(prediction), abs_tol=1e-4)
    # else:
    # prediction = round(prediction, len(str(reference).split(".")[-1]))
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

class TestMathEquality(unittest.TestCase):

    def test_is_numerical_equal(self):
        self.assertTrue(is_numerical_equal("10", "10", True, True))
        self.assertTrue(is_numerical_equal("10", "10", False, True))
        self.assertTrue(is_numerical_equal("10", "10", True, False))
        self.assertFalse(is_numerical_equal("10", "20", True, True))
        self.assertTrue(is_numerical_equal("10%", "0.1", True, True))
        self.assertFalse(is_numerical_equal("10%", "0.2", True, True))

    def test_process_matrix(self):
        prediction, reference = process_matrix("\\begin{pmatrix}1&2\\3&4\\end{pmatrix}", "{1&2,3&4}")
        self.assertEqual(prediction, "\\begin{pmatrix}1&2\\3&4\\end{pmatrix}")
        self.assertEqual(reference, "\\begin{pmatrix}1&2\\3&4\\end{pmatrix}")

    def test_has_format(self):
        self.assertTrue(has_format("[1, 2, 3]", "[", "]"))
        self.assertFalse(has_format("(1, 2, 3)", "[", "]"))

    def test_remove_brackets(self):
        self.assertEqual(remove_brackets("{1, 2, 3}", ["{", "}"]), "1, 2, 3")
        self.assertEqual(remove_brackets("(1, 2, 3)", ["(", ")"]), "1, 2, 3")

    def test_is_list_format(self):
        self.assertTrue(is_list_format("[1, 2, 3]"))
        self.assertTrue(is_list_format("(1, 2, 3)"))
        self.assertFalse(is_list_format("1, 2, 3"))

    def test_is_matrix_format(self):
        self.assertTrue(is_matrix_format("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}"))
        self.assertTrue(is_matrix_format("\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}"))
        self.assertFalse(is_matrix_format("1 2 3 4"))

    def test_compare_lists(self):
        self.assertTrue(compare_lists("[1, 2, 3]", "[1, 2, 3]", True, True))
        self.assertFalse(compare_lists("[1, 2, 3]", "[1, 2, 4]", True, True))
        self.assertTrue(compare_lists("(1, 2, 3)", "(1, 2, 3)", True, True))

    def test_compare_matrices(self):
        self.assertTrue(compare_matrices("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", True, True))
        self.assertFalse(compare_matrices("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", "\\begin{pmatrix}1&2\\\\3&5\\end{pmatrix}", True, True))

    def test_compare_structures(self):
        self.assertTrue(compare_structures("[1, 2, 3]", "[1, 2, 3]", True, True))
        self.assertTrue(compare_structures("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", True, True))
        self.assertFalse(compare_structures("[1, 2, 3]", "[1, 2, 4]", True, True))

    def test_format_equation(self):
        self.assertEqual(format_equation("x = y"), "x - (y)")

    def test_check_symbolic_equality(self):
        self.assertTrue(check_symbolic_equality("x = y", "x = y"))
        self.assertFalse(check_symbolic_equality("x = y", "x = z"))

    def test_check_math_equality(self):
        self.assertTrue(check_math_equality("10", "10", True, True))
        self.assertFalse(check_math_equality("10", "20", True, True))

    def test_is_simple_equation(self):
        self.assertTrue(is_simple_equation("x = 10"))
        self.assertFalse(is_simple_equation("x + y = 10"))

    def test_compare_equations(self):
        self.assertTrue(compare_equations("x = y", "x = y", True, True))
        self.assertTrue(compare_equations("x = 10", "10", True, True))
        self.assertTrue(compare_equations("10", "x = 10", True, True))
        self.assertFalse(compare_equations("x = y", "x = z", True, True))

    def test_math_equal(self):
        self.assertTrue(math_equal("10", "10", True, True))
        self.assertTrue(math_equal("10%", "0.1", True, True))
        self.assertTrue(math_equal("[1, 2, 3]", "[1, 2, 3]", True, True))
        self.assertTrue(math_equal("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", True, True))
        self.assertTrue(math_equal("x = y", "x = y", True, True))
        self.assertFalse(math_equal("10", "20", True, True))
        self.assertFalse(math_equal("x = y", "x = z", True, True))

def _test_math_equal():
    # print(math_equal("0.0833333333333333", "\\frac{1}{12}"))
    # print(math_equal("(1,4.5)", "(1,\\frac{9}{2})"))
    # print(math_equal("\\frac{x}{7}+\\frac{2}{7}", "\\frac{x+2}{7}", timeout=True))
    # print(math_equal("\\sec^2(y)", "\\tan^2(y)+1", timeout=True))
    # print(math_equal("\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\end{pmatrix}", "(\\begin{pmatrix}-\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix})", timeout=True))

    # pred = '\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix})'

    # pred= '-\\frac{8x^2}{9(x^2-2)^{5/3}}+\\frac{2}{3(x^2-2)^{2/3}}'
    # gt= '-\\frac{2(x^2+6)}{9(x^2-2)\\sqrt[3]{x^2-2}^2}'

    # pred =  '-34x-45y+20z-100=0'
    # gt = '34x+45y-20z+100=0'

    # pred = '\\frac{100}{3}'
    # gt = '33.3'

    # pred = '\\begin{pmatrix}0.290243531202435\\\\0.196008371385084\\\\-0.186381278538813\\end{pmatrix}'
    # gt = '(\\begin{pmatrix}0.29\\\\0.196\\\\-0.186\\\\\\end{pmatrix})'

    # pred = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}'
    # gt = '\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}'

    # pred = '(+5)(b+2)'
    # gt = '(a+5)(b+2)'

    # pred = '\\frac{1+\\sqrt{5}}{2}'
    # gt = '2'

    # pred = '\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}', gt = '4'
    # pred = '1', gt = '1\\\\sqrt{19}'

    # pred = "(0.6,2.6667]"
    # gt = "(\\frac{3}{5},\\frac{8}{3}]"

    gt = "x+2n+1"
    pred = "x+1"

    print(math_equal(pred, gt, timeout=True))


if __name__ == "__main__":
    _test_math_equal()
    unittest.main()
