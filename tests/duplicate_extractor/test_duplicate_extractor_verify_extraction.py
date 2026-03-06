from crispen.refactors.duplicate_extractor import _verify_extraction


def test_verify_extraction_valid():
    helper = "def helper(x):\n    return x + 1\n"
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_invalid_helper():
    helper = "def helper(x:\n    pass\n"  # unclosed paren → syntax error after dedent
    replacements = ["result = helper(a)\n"]
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_invalid_replacement():
    helper = "def helper(x):\n    return x\n"
    # Dedented replacement still has a syntax error
    replacements = ["result = helper(a\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False


def test_verify_extraction_no_helper_source():
    # Exercises the helper_source is None branch (skips helper compile check).
    assert _verify_extraction(None, ["result = f()\n"]) is True


def test_verify_extraction_fails_on_param_overwrite():
    # Helper where the parameter is immediately overwritten before being read.
    helper = "def setup(mock_obj):\n    mock_obj = object()\n    return mock_obj\n"
    assert _verify_extraction(helper, ["x = setup(y)\n"]) is False


def test_verify_extraction_allows_return_in_replacement():
    # Replacements inside function bodies legally contain 'return'; the dummy-
    # function wrapper must allow this without triggering a false rejection.
    helper = "def helper(x):\n    return x\n"
    replacements = ["    return helper(a)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_multiline_return_replacement():
    # Multi-line replacement ending with a return statement.
    helper = "def helper(source):\n    return helper(source)\n"
    replacements = [
        "    tree = helper(source)\n    if tree is None:\n        return set()\n"
    ]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_continue_in_replacement():
    # 'continue' is valid inside a loop body; the dummy wrapper now includes a
    # for loop so this is not rejected as a SyntaxError.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        continue\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_break_in_replacement():
    # Same as above but for 'break'.
    helper = "def helper():\n    pass\n"
    replacements = ["    if done:\n        break\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_await_in_replacement():
    # Replacements inside async functions legally contain 'await'; the async
    # dummy-function wrapper must allow this without triggering a false rejection.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_allows_async_helper():
    # async def helpers are valid Python and must compile successfully.
    helper = "async def helper(client, x):\n    return await client.get(x)\n"
    replacements = ["    val = await helper(client, url)\n"]
    assert _verify_extraction(helper, replacements) is True


def test_verify_extraction_rejects_invalid_await_replacement():
    # Replacement with `await` that also has a real syntax error must still fail.
    helper = "async def helper(x):\n    return await x\n"
    replacements = ["    result = await helper(coro\n"]  # unclosed paren
    assert _verify_extraction(helper, replacements) is False
