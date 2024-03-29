""" Contains the project metadata,
    e.g., *title*, *version*, *summary* etc.
"""

from datetime import datetime

__MAJOR__ = 0
__MINOR__ = 1
__PATCH__ = 0

__title__ = "text_multiclass_classification"
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = "A multi-class text classification framework with PyTorch."
__author__ = "Konstantinos Kanaris"
__copyright__ = f"Copyright (C) {datetime.now().date().year}  {__author__}"
__email__ = "konskan95@outlook.com.gr"
