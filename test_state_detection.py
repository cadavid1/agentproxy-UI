"""
Tests for Gemini-based state detection functions.

These tests verify that:
1. _looks_like_done() correctly identifies task completion via Gemini
2. _review_has_issues() correctly identifies code review issues via Gemini
"""

import unittest
from pa import FileChangeTracker, FunctionExecutor, GeminiClient


class TestLooksLikeDone(unittest.TestCase):
    """Test the _looks_like_done method that uses Gemini for semantic analysis."""
    
    def setUp(self):
        self.tracker = FileChangeTracker(".")
    
    def test_empty_text_returns_false(self):
        """Empty or very short text should return False without calling Gemini."""
        self.assertFalse(self.tracker._looks_like_done(""))
        self.assertFalse(self.tracker._looks_like_done("   "))
        self.assertFalse(self.tracker._looks_like_done("hi"))
    
    def test_clear_completion_message(self):
        """Clear task completion messages should return True."""
        completion_messages = [
            "I have completed the task. The calculator.py file has been created with add, subtract, multiply, and divide functions.",
            "Task complete! I've implemented all the requested features and verified they work correctly.",
            "Done! The Flask server is now running on port 5000 with all endpoints functional.",
            "I've finished implementing the requested changes. All tests pass.",
            "Successfully created the hello.py file with the greeting function as requested.",
        ]
        
        for msg in completion_messages:
            with self.subTest(msg=msg[:50]):
                result = self.tracker._looks_like_done(msg)
                self.assertTrue(result, f"Expected True for: {msg[:50]}...")
    
    def test_ongoing_work_message(self):
        """Messages indicating ongoing work should return False."""
        ongoing_messages = [
            "Let me read the file first to understand the current implementation.",
            "I'm going to create a new function to handle this case.",
            "Now I need to fix the bug in the authentication module.",
            "I'll start by setting up the project structure.",
            "Looking at the error, I think we need to modify the database schema.",
        ]
        
        for msg in ongoing_messages:
            with self.subTest(msg=msg[:50]):
                result = self.tracker._looks_like_done(msg)
                self.assertFalse(result, f"Expected False for: {msg[:50]}...")
    
    def test_error_message(self):
        """Error messages should return False."""
        error_messages = [
            "Error: The file could not be found. Let me check the path.",
            "There's a syntax error on line 45. I need to fix this.",
            "The test failed because the expected output doesn't match.",
        ]
        
        for msg in error_messages:
            with self.subTest(msg=msg[:50]):
                result = self.tracker._looks_like_done(msg)
                self.assertFalse(result, f"Expected False for: {msg[:50]}...")
    
    def test_question_message(self):
        """Questions from Claude should return False."""
        question_messages = [
            "Should I also add error handling to this function?",
            "Do you want me to create unit tests for this module?",
            "Which database would you prefer - PostgreSQL or SQLite?",
        ]
        
        for msg in question_messages:
            with self.subTest(msg=msg[:50]):
                result = self.tracker._looks_like_done(msg)
                self.assertFalse(result, f"Expected False for: {msg[:50]}...")


class TestReviewHasIssues(unittest.TestCase):
    """Test the _review_has_issues method that uses Gemini for semantic analysis."""
    
    def setUp(self):
        self.executor = FunctionExecutor(".")
    
    def test_empty_review_returns_false(self):
        """Empty or very short review should return False."""
        self.assertFalse(self.executor._review_has_issues(""))
        self.assertFalse(self.executor._review_has_issues("   "))
        self.assertFalse(self.executor._review_has_issues("ok"))
    
    def test_passing_review(self):
        """Reviews that pass should return False (no issues)."""
        passing_reviews = [
            """## ERRORS
- None

## ISSUES
- None

## IMPROVEMENTS
- Code looks good

## VERDICT
[PASS] - Code is correct and well-structured""",
            
            """NO ISSUES FOUND

The code is clean, well-documented, and follows best practices.
All functions have proper error handling.""",
            
            """## ERRORS
- None

## ISSUES  
- None

## IMPROVEMENTS
- Consider adding type hints (minor suggestion)

## VERDICT
[PASS] - Implementation is correct""",
        ]
        
        for review in passing_reviews:
            with self.subTest(review=review[:50]):
                result = self.executor._review_has_issues(review)
                self.assertFalse(result, f"Expected False (no issues) for passing review")
    
    def test_failing_review_with_errors(self):
        """Reviews with errors should return True (has issues)."""
        failing_reviews = [
            """## ERRORS
- Line 15: SyntaxError - missing closing parenthesis
- Line 28: NameError - variable 'result' is not defined

## ISSUES
- Function divide() doesn't handle division by zero

## VERDICT
[FAIL] - Critical errors must be fixed""",
            
            """## ERRORS
- The import statement on line 3 will fail because 'numpy' is not installed

## ISSUES
- Security vulnerability: SQL injection possible on line 45

## VERDICT
[FAIL] - Security issue and import error""",
            
            """BUG FOUND: The loop condition on line 20 will cause an infinite loop.
This will crash the application.""",
        ]
        
        for review in failing_reviews:
            with self.subTest(review=review[:50]):
                result = self.executor._review_has_issues(review)
                self.assertTrue(result, f"Expected True (has issues) for failing review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
