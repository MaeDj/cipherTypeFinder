#Single source of truth for the binarization method prompt/list, shared by every pipeline
#stage that needs to know which binarization method's output to read/write.
#main.py asks once and forwards the resolved method name to every stage (as a plain
#CLI arg to the ones it launches as subprocesses); each stage still falls back to its
#own interactive prompt when run standalone (no arg given).
BINARIZATION_METHODS = ["otsu", "gauss", "adaptive", "niblack", "sauvola", "local"]


def resolve_binarization_method(preselected=None):
    """Return the binarization method name (e.g. "sauvola").

    preselected: a method name already chosen upstream (e.g. sys.argv[1] forwarded by
    main_preprocessing.py). When given, it's validated and returned as-is - no prompt.
    Otherwise, interactively asks once, same prompt/default every stage used to repeat.
    """
    if preselected:
        if preselected not in BINARIZATION_METHODS:
            raise ValueError(
                f"Unknown binarization method '{preselected}'. Expected one of {BINARIZATION_METHODS}"
            )
        return preselected

    choice = False
    while choice not in ["", "1", "2", "3", "4", "5", "6"]:
        choice = input(
            "Select the binarization method to use: [Default:5] "
            "(1:Otsu 2:Gaussian 3:Adaptive 4:Niblack 5:Sauvola 6:Local) \n"
        )
    if choice == "":
        choice = "5"

    return BINARIZATION_METHODS[int(choice) - 1]