
# class ProcessCorrData(SingleTask):
#     _file_index = 0

#     def setup(self, file_list):
#         self.files = file_list

#     def process(self):
#         if self._file_index == len(self.files):
#             raise pipeline.PipelineStopIteration
#         template_ts = TimeStream.from_file(
#             self.files[self._file_index], distributed=True
#         )
#         # debug
#         self.log.info(f"template_ts.vis.shape: {template_ts.vis[:].shape}")
#         if hasattr(template_ts, 'index_map'):
#             for axis_name in ['time', 'freq', 'prod', 'input']:
#                 if axis_name in template_ts.index_map:
#                     axis_len = len(template_ts.index_map[axis_name])
#                     self.log.info(f"index_map['{axis_name}'] length: {axis_len}")

#         dist_ts = TimeStream(
#             data=MPIArray.from_hdf5(self.files[self._file_index], "vis", axis=0),
#             axes_from=template_ts,
#         )
#         # Move axes from (time, freq, bl) to (bl, time, freq)
#         dist_ts.vis[:] = np.moveaxis(template_ts.vis[:], (0, 1, 2), (2, 0, 1))[dist_ts.vis[:].local_bounds, ...]

#         template_wt = MPIArray.from_hdf5(
#             self.files[self._file_index], 
#             "flags/vis_weight", 
#             axis=0
#         )
#         transformed_wt = np.moveaxis(template_wt, (0, 1, 2), (2, 0, 1))
#         # Assign to the distributed container
#         # First check if vis_weight exists, if not create it
#         if 'vis_weight' not in dist_ts:
#             dist_ts.create_dataset(
#                 'vis_weight',
#                 data=transformed_wt[dist_ts.vis[:].local_bounds, ...],
#                 distributed=True,
#                 distributed_axis=0
#             )
#         else:
#             dist_ts['vis_weight'][:] = transformed_wt[dist_ts.vis[:].local_bounds, ...]
        
        
#         vis = dist_ts.vis[:]
#         wt = dist_ts.weight[:]
#         max_weight = 1e10
#         min_weight = 1e-10
#         bads = ~((np.isfinite(wt)) & (np.isfinite(vis)) & (wt > min_weight) & (wt < max_weight))
#         dist_ts.weight[:][bads] = 0
#         dist_ts.vis[:][bads] = 0
#         self._file_index += 1
#         return dist_ts