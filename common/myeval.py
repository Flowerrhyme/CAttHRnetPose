import numpy as np
from common.general import box_iou
import copy

IMG_H=640
IMG_W=640



class Eval:
    def __init__(self, gts, pds, iouType = None, txtname=None, dis = False): # gts is list for many images
        self.finename = txtname
        self.gts = []
        self.pds = []
        for i,gt in enumerate(gts): #list
            tmps = []
            for j,g in enumerate(gt.cpu()): #tensor
                temp = {}
                temp['area'] = g[2]*g[3]*IMG_H*IMG_W  #w*h
                temp['id']=str(i)+str(j)
                temp['bbox'] = g[:4]*IMG_W
                temp['keypoints'] = g[4:]*IMG_W
                tmps.append(temp)
            self.gts.append(tmps)
        for i,pd in enumerate(pds): #list
            tmps = []
            for j,p in enumerate(pd.cpu()): #tensor
                temp = {}
                temp['area'] = p[2]*p[3]  #w*h
                temp['boxscore'] = p[4]
                temp['kptscore'] = p[7::3]
                temp['score'] = temp['kptscore'].mean()*temp['boxscore']
                temp['id']=int(str(i)+str(j))
                temp['bbox'] = p[:4]
                temp['keypoints'] = p[5:]
                tmps.append(temp)
            self.pds.append(tmps)
        self.params = Params(iouType=iouType)
        if iouType == 'bbox':
            self.compute = self.computeIoU
        elif iouType == 'keypoints':
            self.compute = self.computeOks
        self.compute_dis = dis
        self.iouType = iouType

    def setiou(self, iouType):
        if iouType == 'bbox':
            self.compute = self.computeIoU
        elif iouType == 'keypoints':
            self.compute = self.computeOks

    def evaluate(self):
        p = self.params
        maxDet = p.maxDets[-1]
        '''self.ious = {i: self.computeIoU(gt, pd) \
                        for i,(gt,pd) in enumerate(zip(self.gts,self.pds)) }'''
        self.ious = {i: self.compute(gt, pd) \
                        for i,(gt,pd) in enumerate(zip(self.gts,self.pds)) }
        if self.compute_dis:
            self.dis = [self.computeDis(gt, pd) \
                            for i,(gt,pd) in enumerate(zip(self.gts,self.pds)) ]

        self.evalImgs = [self.evaluateImg(imgId, areaRng, maxDet)
                         for areaRng in p.areaRng
                         for imgId in range(len(self.gts))]

    def computeIoU(self, gt, dt):
        p = self.params

        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        ious = box_iou(d,g)
        return ious
    
    def computeOks(self, gts, dts):
        p = self.params
        # dimention here should be Nxm
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::2]; yg = g[1::2]
            k1 = np.count_nonzero((xg > 0) & (yg > 0))
            #mask = np.array((xg != 0) & (yg != 0))
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]

                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = (xd - xg) #*mask
                    dy = (yd - yg) #*mask
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[(xg > 0) & (yg > 0)].numpy()
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious
    
    def computeDis(self,gts,dts):
        p = self.params
        # dimention here should be Nxm
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.ones((3,len(gts),17))*np.inf
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::2]; yg = g[1::2]
            mask = np.array((xg != 0) & (yg != 0))

            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]

                dx = abs((xd - xg)*mask)
                dy = abs((yd - yg)*mask)

                ious[0,j] = dx if dx.mean()<ious[0,j].mean() else ious[0,j]
                ious[1,j] = dy if dy.mean()<ious[1,j].mean() else ious[1,j]
        return np.mean(ious,axis=1)

    def evaluateImg(self, imgId, aRng, maxDet):
        
        gt = self.gts[imgId]
        dt = self.pds[imgId]

        for g in gt:
            if (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        #sort 
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]

        ious = self.ious[imgId][:, gtind] if len(self.ious[imgId]) > 0 else self.ious[imgId]

        p = self.params
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtIg = np.array([g['_ignore'] for g in gt])
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        if gtm[tind,gind]>0:
                            continue
                        if ious[dind,gind] < iou:
                            continue
                        iou=ious[dind,gind]
                        m=gind
                    if m ==-1:
                        continue
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        return {
                'image_id':     imgId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self):           
        p = self.params    
        self._paramsEval = copy.deepcopy(self.params) 
        _pe = self._paramsEval

        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))


        m_list = [m for n, m in enumerate(p.maxDets)]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))]
        i_list = [n for n, i in enumerate(range(len(self.gts)))]

        I0 = len(self.gts)
        A0 = len(_pe.areaRng)

        
        for a, a0 in enumerate(a_list):
            Na = a0*I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Na + i] for i in i_list]
                E = [e for e in E if not e is None]
                if len(E) == 0:
                    continue
                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.
                inds = np.argsort(-dtScores, kind='mergesort')
                dtScoresSorted = dtScores[inds]

                dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg==0 )
                if npig == 0:
                    continue
                tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    nd = len(tp)
                    rc = tp / npig
                    pr = tp / (fp+tp+np.spacing(1))
                    q  = np.zeros((R,))
                    ss = np.zeros((R,))

                    if nd:
                        recall[t,0,a,m] = rc[-1]
                    else:
                        recall[t,0,a,m] = 0

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    pr = pr.tolist(); q = q.tolist()

                    for i in range(nd-1, 0, -1):
                        if pr[i] > pr[i-1]:
                            pr[i-1] = pr[i]

                    inds = np.searchsorted(rc, p.recThrs, side='left')
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = pr[pi]
                            ss[ri] = dtScoresSorted[pi]
                    except:
                        pass
                    precision[t,:,0,a,m] = np.array(q)
                    scores[t,:,0,a,m] = np.array(ss)        

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _dis_summarize():
            mdx = np.zeros(17)
            mdy = np.zeros(17)
            count = np.zeros(17)
            for dis in self.dis:
                if len(dis):
                    mdx += dis[0]
                    count += (dis[0]!=0)
                    mdy += dis[1]
            mdx /= count
            mdy /= count
            print(mdx)
            print(mdy)

                


        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if self.finename is None:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            else:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
                with open(self.finename+'test.txt','a') as f:
                    f.write(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)+'\n')

            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        
        if self.compute_dis:
            _dis_summarize()
        if self.iouType == 'bbox':
            summarize = _summarizeDets
        elif self.iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        

    def __str__(self):
        self.summarize()




class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
    def setKpParams(self):
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

    def __init__(self, iouType='keypoints'):
        if iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType