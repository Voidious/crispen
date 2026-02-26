def check_permission(user):
    if not user.is_staff:
        return "forbidden"
    else:
        return "allowed"


def check_ownership(resource, user):
    if resource.owner == user:
        return "owner"
    else:
        return "not owner"
