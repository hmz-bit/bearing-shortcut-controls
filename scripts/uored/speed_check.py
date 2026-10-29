import numpy as np, glob, os
from scipy import signal
out=[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data','uored','csv','*.csv'))):
    a=np.loadtxt(f,delimiter=',',skiprows=1,usecols=(0,2),encoding='utf-8-sig')
    x=a[:,0]-a[:,0].mean(); rec=a[:,1].max()/60.0
    # Welch with 4 s segments (0.25 Hz resolution) to be robust
    fr,P=signal.welch(x,fs=42000,nperseg=168000,window='hann')
    sel=(fr>=20)&(fr<=40)
    pk=fr[sel][np.argmax(P[sel])]
    # prominence of the peak vs neighbourhood median
    med=np.median(P[(fr>=15)&(fr<=45)])
    out.append((os.path.basename(f),rec,pk,100*(pk-rec)/rec,P[sel].max()/med))
print("file rec_fr(Hz) peak_fr(Hz) diff% prominence")
for o in out: print("%-12s %7.2f %7.2f %6.2f %7.1f"%o)
d=np.array([abs(o[3]) for o in out]); print("files with |diff|<1.5%%: %d of %d; <5%%: %d"%((d<1.5).sum(),len(d),(d<5).sum()))
