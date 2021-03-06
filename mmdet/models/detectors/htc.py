import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from mmcv.runner import get_dist_info

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .. import builder
from ..registry import DETECTORS
from .cascade_rcnn import CascadeRCNN

import os
import errno

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

CAT2LABEL_PATH = './cat2label.tmp'

TOTAL_NUM = 5000   # number of iter for phase 3 (LOAD_GT_DIST)
INDICES_PATH = './data/LVIS/lvis_step1_320_sorted/lvis_indices_qry_step1_rand_balanced.json'
CLASS_PATH = './data/LVIS/lvis_step1_320_sorted/lvis_classes_qry_step1_rand_balanced.json'
SAVE_PATH = '/data1/lvis_test1/'
ALL_DIST_PATH = '/data1/lvis_test1/all.dist'
PREV_DIM = 270

SAVE_GT_BOX = False    # phase 1
SAVE_LOGITS = False     # phase 2
LOAD_GT_DIST = False      # phase 3


##############################
# 0.0 change INDICES_PATH, CLASS_PATH, SAVE_PATH, ALL_DIST_PATH, PREV_DIM
#
# 1.1 (1) change (num_classes=num_old + 320, ann_file, img_prefix) in htc_x101_64x4d_fpn_20e_16gpu_0.py; (2) SAVE_GT_BOX=True
# 1.2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=112211 ./tools/dist_train.sh configs/htc/htc_x101_64x4d_fpn_20e_16gpu_0.py 8
# 
# 2.1 (1) change (num_classes=num_old, ann_file, img_prefix) in htc_x101_64x4d_fpn_20e_16gpu_1.py; (2) SAVE_LOGITS=True
# 2.2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=112222 ./tools/dist_test.sh configs/htc/htc_x101_64x4d_fpn_20e_16gpu_1.py ./work_dirs/270_x64/epoch_20_no_bias.pth 8 --out ./work_dirs/stage2_tmp/tmp.pkl --eval bbox segm
# 2.3 merge_dist_files.py  automatically merge all generated distill logits
#
# 3.1 update_checkpoint.py (need to change top 4 parameters in the file)  to expand weight from old_dim -> old + 320 
# 3.2 (1) change (num_classes=num_old + 320, ann_file, img_prefix, total_epochs+1, step + 20k/35k) in htc_x101_64x4d_fpn_20e_16gpu_2.py; (2) LOAD_GT_DIST=True, 
# 3.3 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=112233 ./tools/dist_train.sh configs/htc/htc_x101_64x4d_fpn_20e_16gpu_2.py 8 --validate --resume_from ./work_dirs/270_x64/epoch_20_pad.pth
##############################

@DETECTORS.register_module
class HybridTaskCascade(CascadeRCNN):

    def __init__(self,
                 num_stages,
                 backbone,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 **kwargs):
        super(HybridTaskCascade, self).__init__(num_stages, backbone, **kwargs)
        assert self.with_bbox and self.with_mask
        assert not self.with_shared_head  # shared head not supported
        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.interleaved = interleaved
        self.mask_info_flow = mask_info_flow
        self.distill_loss = nn.MSELoss(reduction='sum')

        rank, _ = get_dist_info()
        self.class_idx = json.load(open(CLASS_PATH))[rank]
        self.train_iter = 0
        self.cat2label = None

        if LOAD_GT_DIST:
            self.loaded_gt_dist = torch.load(ALL_DIST_PATH)
            self.loaded_gt_keys = list(self.loaded_gt_dist.keys())

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat

        cls_score, bbox_pred = bbox_head(bbox_feats, norm_on=True)
        self.num_dist_cls = cls_score.shape[-1]
        bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                            gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)

        #########################################################
        if LOAD_GT_DIST:
            gt_rois = bbox2roi(gt_bboxes)
            gt_len = [len(b) for b in gt_bboxes]
            gt_bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        gt_rois)
            gt_cls_score, _ = bbox_head(gt_bbox_feats, norm_on=True, scale=False)
            self.pred_new_dist[stage] = gt_cls_score[:,1:PREV_DIM+1].split(gt_len, dim=0)
        #########################################################

        return loss_bbox, rois, bbox_targets, bbox_pred

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        pos_rois)

        # semantic feature fusion
        # element-wise sum for original features and pooled semantic features
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             pos_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat

        # mask information flow
        # forward all previous mask heads to obtain last_feat, and fuse it
        # with the normal mask feature
        if self.mask_info_flow:
            last_feat = None
            for i in range(stage):
                last_feat = self.mask_head[i](
                    mask_feats, last_feat, return_logits=False)
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
        else:
            mask_pred = mask_head(mask_feats, return_feat=False)

        mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                            rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
        return loss_mask

    def _bbox_forward_test(self, stage, x, rois, semantic_feat=None):
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            bbox_feats += bbox_semantic_feat
       
        ##############################################################
        if SAVE_LOGITS:
            cls_score, bbox_pred = bbox_head(bbox_feats, norm_on=True, scale=False)
            self.output_logits_dict[stage] = cls_score.split(self.gt_length, dim=0)
        else:
            cls_score, bbox_pred = bbox_head(bbox_feats, norm_on=True)
        ##############################################################
        
        self.num_dist_cls = cls_score.shape[-1]
        return cls_score, bbox_pred

    def _mask_forward_test(self, stage, x, bboxes, semantic_feat=None):
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_rois = bbox2roi([bboxes])
        mask_feats = mask_roi_extractor(
            x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            mask_feats += mask_semantic_feat
        if self.mask_info_flow:
            last_feat = None
            last_pred = None
            for i in range(stage):
                mask_pred, last_feat = self.mask_head[i](mask_feats, last_feat)
                if last_pred is not None:
                    mask_pred = mask_pred + last_pred
                last_pred = mask_pred
            mask_pred = mask_head(mask_feats, last_feat, return_feat=False)
            if last_pred is not None:
                mask_pred = mask_pred + last_pred
        else:
            mask_pred = mask_head(mask_feats)
        return mask_pred

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # semantic head
        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None
        # bbox heads
        rois = bbox2roi([proposals])
        for i in range(self.num_stages):
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            mask_roi_extractor = self.mask_roi_extractor[-1]
            mask_feats = mask_roi_extractor(
                x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_semantic and 'mask' in self.semantic_fusion:
                mask_semantic_feat = self.semantic_roi_extractor(
                    [semantic_feat], mask_rois)
                mask_feats += mask_semantic_feat
            last_feat = None
            for i in range(self.num_stages):
                mask_head = self.mask_head[i]
                if self.mask_info_flow:
                    mask_pred, last_feat = mask_head(mask_feats, last_feat)
                else:
                    mask_pred = mask_head(mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None):

        #####################################################
        # 1. change train/test random_flip = 0.0,    epoch = 1,    change test date to train set， frozen_stages = 3， rcnn.num = 200
        # 2. train -> save gt box

        # create folder if not exist
        if not os.path.exists(SAVE_PATH):
            mkdir(SAVE_PATH)

        # save each gt box info
        # meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg')
        # coco train/val/test size (118287, 5000, 40670)
        if SAVE_GT_BOX:
            for meta, gt_bx in zip(img_meta, gt_bboxes):
                file_name = meta['filename'].split('/')[-1].split('.')[0]
                output_file = SAVE_PATH + file_name
                output_dict = {'gt_bbox': gt_bx.cpu()}
                torch.save(output_dict, output_file)            


        # load dist of gt boxes
        select_class = None
        if LOAD_GT_DIST:
            self.pred_new_dist = {}
            self.is_empty = []
            for meta, gt_b in zip(img_meta, gt_bboxes):
                file_name = meta['filename'].split('/')[-1].split('.')[0]
                if (file_name in self.loaded_gt_keys) and (gt_b.shape[0] > 0):
                    self.is_empty.append(False)
                else:
                    self.is_empty.append(True)

            if self.cat2label is None:
                self.cat2label = torch.load(CAT2LABEL_PATH)

            select_class = self.cat2label[self.class_idx[self.train_iter]]
            self.train_iter += 1

        ######################################################

        x = self.extract_feat(img)

        losses = dict()

        # RPN part, the same as normal two-stage detectors
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # semantic segmentation part
        # 2 outputs: segmentation prediction and embedded features
        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
            bbox_sampler = build_sampler(rcnn_train_cfg.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    select_label=select_class,
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            # bbox head forward and loss
            loss_bbox, rois, bbox_targets, bbox_pred = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat)
            roi_labels = bbox_targets[0]

            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                # interleaved execution: use regressed bboxes by the box branch
                # to train the mask branch
                if self.interleaved:
                    pos_is_gts = [res.pos_is_gt for res in sampling_results]
                    with torch.no_grad():
                        proposal_list = self.bbox_head[i].refine_bboxes(
                            rois, roi_labels, bbox_pred, pos_is_gts, img_meta)
                        # re-assign and sample 512 RoIs from 512 RoIs
                        sampling_results = []
                        for j in range(num_imgs):
                            assign_result = bbox_assigner.assign(
                                proposal_list[j], gt_bboxes[j],
                                gt_bboxes_ignore[j], gt_labels[j])
                            sampling_result = bbox_sampler.sample(
                                assign_result,
                                proposal_list[j],
                                gt_bboxes[j],
                                gt_labels[j],
                                feats=[lvl_feat[j][None] for lvl_feat in x])
                            sampling_results.append(sampling_result)
                loss_mask = self._mask_forward_train(i, x, sampling_results,
                                                     gt_masks, rcnn_train_cfg,
                                                     semantic_feat)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes (same as Cascade R-CNN)
            if i < self.num_stages - 1 and not self.interleaved:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta)

        ##########################################
        if LOAD_GT_DIST:
            distilation_gt_dist = []
            distilation_pd_dist = []
            for i, meta in enumerate(img_meta):
                file_name = meta['filename'].split('/')[-1].split('.')[0]
                if self.is_empty[i]:
                    continue
                else:
                    for s in range(self.num_stages):
                        distilation_gt_dist.append(self.loaded_gt_dist[file_name][s])
                        distilation_pd_dist.append(self.pred_new_dist[s][i])
            if len(distilation_gt_dist) > 0 :
                distilation_pd_dist = torch.cat(distilation_pd_dist, dim=0)
                distilation_gt_dist = torch.cat(distilation_gt_dist, dim=0)[:, 1:].to(distilation_pd_dist.device)
                if distilation_pd_dist.shape[0] == distilation_gt_dist.shape[0]:
                    losses['distill_loss'] = self.distill_loss(distilation_pd_dist, distilation_gt_dist) * self.num_stages / (distilation_pd_dist.shape[0] + 1e-9)
                else:
                    print('distill loss: gt vs. pred size not match')
                    losses['distill_loss'] = torch.zeros([1]).to(x[0].device)
            else:
                print('distill loss: gt missing')
                losses['distill_loss'] = torch.zeros([1]).to(x[0].device)
        ##########################################

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        ###############################################################
        # 1. change train/test random_flip = 0.0,    epoch = 1,    change test date to train set
        # 2. test  -> use gt box to extract gt_dist

        # create folder if not exist
        if not os.path.exists(SAVE_PATH):
            mkdir(SAVE_PATH)

        if SAVE_LOGITS:
            self.output_logits_dict = {}
            self.gt_length = []
            self.is_empty = []
            gt_proposals = []
            for meta, prop in zip(img_meta, proposal_list):
                file_name = meta['filename'].split('/')[-1].split('.')[0]
                if os.path.exists(SAVE_PATH + file_name):
                    gt_prop = torch.load(SAVE_PATH + file_name)['gt_bbox'].to(prop.device)
                    gt_proposals.append(gt_prop)
                    self.gt_length.append(int(gt_prop.shape[0]))
                    self.is_empty.append(False)
                else:
                    gt_proposals.append(torch.zeros(1,4).to(prop.device))
                    self.gt_length.append(1)
                    self.is_empty.append(True)

            proposal_list = gt_proposals
        #######################################################

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            cls_score, bbox_pred = self._bbox_forward_test(
                i, x, rois, semantic_feat=semantic_feat)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0])

        cls_score = sum(ms_scores) / float(len(ms_scores))
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                mask_roi_extractor = self.mask_roi_extractor[-1]
                mask_feats = mask_roi_extractor(
                    x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    mask_feats += mask_semantic_feat
                last_feat = None
                for i in range(self.num_stages):
                    mask_head = self.mask_head[i]
                    if self.mask_info_flow:
                        mask_pred, last_feat = mask_head(mask_feats, last_feat)
                    else:
                        mask_pred = mask_head(mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if self.with_mask:
            results = (ms_bbox_result['ensemble'], ms_segm_result['ensemble'])
        else:
            results = ms_bbox_result['ensemble']

        #####################################################
        if SAVE_LOGITS:
            for i, (meta, empty) in enumerate(zip(img_meta, self.is_empty)):
                file_name = meta['filename'].split('/')[-1].split('.')[0]
                output_dict = {}
                for key, val in self.output_logits_dict.items():
                    if not empty:
                        output_dict[key] = val[i].cpu()
                    else:
                        output_dict[key] = torch.zeros(0, self.num_dist_cls)
                output_file = SAVE_PATH + file_name + '.dist'
                torch.save(output_dict, output_file)
        #####################################################

        return results

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1]
                for feat in self.extract_feats(imgs)
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(
                self.extract_feats(imgs), img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                cls_score, bbox_pred = self._bbox_forward_test(
                    i, x, rois, semantic_feat=semantic)
                ms_scores.append(cls_score)

                if i < self.num_stages - 1:
                    bbox_label = cls_score.argmax(dim=1)
                    rois = bbox_head.regress_by_class(rois, bbox_label,
                                                      bbox_pred, img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes -
                                              1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(
                        self.extract_feats(imgs), img_metas, semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_rois = bbox2roi([_bboxes])
                    mask_feats = self.mask_roi_extractor[-1](
                        x[:len(self.mask_roi_extractor[-1].featmap_strides)],
                        mask_rois)
                    if self.with_semantic:
                        semantic_feat = semantic
                        mask_semantic_feat = self.semantic_roi_extractor(
                            [semantic_feat], mask_rois)
                        if mask_semantic_feat.shape[-2:] != mask_feats.shape[
                                -2:]:
                            mask_semantic_feat = F.adaptive_avg_pool2d(
                                mask_semantic_feat, mask_feats.shape[-2:])
                        mask_feats += mask_semantic_feat
                    last_feat = None
                    for i in range(self.num_stages):
                        mask_head = self.mask_head[i]
                        if self.mask_info_flow:
                            mask_pred, last_feat = mask_head(
                                mask_feats, last_feat)
                        else:
                            mask_pred = mask_head(mask_feats)
                        aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
            return bbox_result, segm_result
        else:
            return bbox_result
