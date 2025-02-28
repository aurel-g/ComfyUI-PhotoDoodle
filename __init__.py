from .PhotoDoodle_nodes import PhotoDoodle

NODE_CLASS_MAPPINGS = {
    "PhotoDoodle Gen": PhotoDoodle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoodle": "PhotoDoodle Gen",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
