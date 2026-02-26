def check_permission(user):
    if not user.is_staff:
        return "forbidden"
    else:
        return "allowed"


def check_ownership(resource, user):
    if not resource.owner == user:
        return "not owner"
    else:
        return "owner"
