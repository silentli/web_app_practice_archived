class UserInputError(ValueError):
    """exception raised when the user provides invalid or incomplete input"""
    pass

class InternalProcessingError(ValueError):
    """exception raised when an unexpected issue occurs during internal processing"""
    pass
