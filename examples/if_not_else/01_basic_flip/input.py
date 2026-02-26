# A simple request handler where the guard condition is negated.
# Crispen should flip it to the positive form.
def handle_request(request):
    if not request.is_authenticated:
        return unauthorized_response()
    else:
        return process_request(request)
