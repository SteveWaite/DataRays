import bk_io
import os
import tensorflow as tf
from typing import List, Optional, Union


class Checkpointer:
    def __init__(self, file_format: str):
        '''
            Args:
                file_format: format string with variable {checkpoint_i}.
                    Ex. '/my/path/{checkpoint_i}.checkpoint'
        '''
        super().__init__()
        self.file_format = file_format

    def path_for_checkpoint(self, checkpoint_i: Union[int, str]) -> str:
        return self.file_format.format(checkpoint_i=checkpoint_i)

    def list_checkpoints(self) -> List[int]:
        file_names = tf.io.gfile.glob(self.path_for_checkpoint('*'))

        def file_name_to_checkpoint_index(file_name: str) -> Optional[int]:
            try:
                return int(os.path.splitext(os.path.basename(file_name))[0])
            except ValueError:
                return None

        checkpoint_is = list(
            filter(
                lambda i: i is not None,
                map(file_name_to_checkpoint_index, file_names)))
        checkpoint_is.sort()
        return checkpoint_is

    def save_checkpoint(self, checkpoint_i: int, content: bytes) -> None:
        bk_io.write_binary_file(self.path_for_checkpoint(checkpoint_i), content)

    def load_checkpoint(self, checkpoint_i: int) -> bytes:
        return bk_io.read_binary_file(self.path_for_checkpoint(checkpoint_i))
