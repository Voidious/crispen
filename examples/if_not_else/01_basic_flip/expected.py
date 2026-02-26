# A simple request handler where the guard condition is negated.
# Crispen should flip it to the positive form.
def handle_request(request):
    if request.is_authenticated:
        return process_request(request)
    else:
        return unauthorized_response()
