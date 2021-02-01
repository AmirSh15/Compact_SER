import numpy as np

tr_feat = np.load('../dataset/IEMOCAP_data.npy')
tr_label = np.load('../dataset/IEMOCAP_label.npy')


txt = []

txt.append(tr_feat.shape[0])
for i in range(tr_feat.shape[0]):
    for j in range(tr_feat.shape[1]):
        if(j==0):
            txt.append('%s %s'% (tr_feat.shape[1], int(tr_label[i])))
        if(j==tr_feat.shape[1]-1    ):
            a = [e.astype('float16') for e in tr_feat[i, j]]
            txt.append('%s 1 %s ' % (j, j -1) + ' '.join(str(e) + ' ' for e in a))
        else:
            a =[e.astype('float16') for e in tr_feat[i,j]]
            txt.append('%s 1 %s '% (j, j+1)+ ' '.join(str(e)+' ' for e in a))


np.savetxt('IEMOCAP.txt', txt, fmt='%s')
