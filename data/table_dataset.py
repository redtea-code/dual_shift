"""
Table-only dataset: wraps FullDatasetClassifier, replaces MRI with dummy zeros.
MRI-table pairing, subject extraction, table joining — all via parent class.
"""
import os
import torch
import numpy as np

from data.dataset_v2 import FullDatasetClassifier, create_dataset


class TableOnlyDataset(FullDatasetClassifier):
    """
    Same as FullDatasetClassifier but __getitem__ uses a zero-filled
    dummy tensor instead of loading real MRI images.

    MRI-table pairing is handled by FullDatasetClassifier.__init__
    via save_load=True → nii.gz → pt conversion + _find_index matching.
    """

    def __getitem__(self, index: int):
        folder = self._folder_names[index]
        pt_path = self.PET_pt[index]

        # lazy-detect image shape from first sample
        if not hasattr(self, '_dummy_shape'):
            sample = torch.load(pt_path, map_location='cpu')
            self._dummy_shape = sample.shape

        image = torch.zeros(self._dummy_shape, dtype=torch.float32)

        batch = {
            'image': image,
            'label': int(self.all_labels[index] + 1),  # 1-based
            'subject_id': self._subject_ids[index],
            'folder': folder,
        }

        if self.import_table and self.table_df_prepared is not None:
            _, date_index = self._find_index(folder, self.table_df)
            if date_index >= 0:
                cat_vals = self.table_df_prepared['cate_x'].iloc[date_index].values
                cont_vals = self.table_df_prepared['conti_x'].iloc[date_index].values
                # Ensure at least 1-dim for stacking (empty cat/cont causes batch mismatch)
                batch['cate_x'] = torch.tensor(cat_vals if len(cat_vals) > 0 else [0], dtype=torch.int64)
                batch['conti_x'] = torch.tensor(cont_vals if len(cont_vals) > 0 else [0.0], dtype=torch.float32)
            else:
                cat0 = self.table_df_prepared['cate_x'].iloc[0].values
                cont0 = self.table_df_prepared['conti_x'].iloc[0].values
                batch['cate_x'] = torch.zeros(max(len(cat0), 1), dtype=torch.int64)
                batch['conti_x'] = torch.zeros(max(len(cont0), 1), dtype=torch.float32)

        return batch


def load_table_dataset(cf: dict) -> TableOnlyDataset:
    """Mirror of load_v4_dataset but returns TableOnlyDataset."""
    data_root = cf['data_root']
    # Always use *.pt — create_dataset auto-converts nii.gz if needed
    file_pattern = '*.pt'
    table_path = cf.get('table_path', '')

    if not table_path or not os.path.isfile(table_path):
        table_path = ''

    ds = create_dataset(
        data_root=data_root,
        table_path=table_path,
        file_pattern=file_pattern,
        desired_shape=cf.get('img_sz', (160, 160, 96)),
        days_threshold=cf.get('days_threshold', 90),
        dataset=cf.get('dataset', 'NACC'),
    )

    # Swap class — keep all data, just override __getitem__
    ds.__class__ = TableOnlyDataset
    return ds
