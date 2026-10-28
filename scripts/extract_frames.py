"""Extract frames from a consented video into a chosen split/class folder."""
import argparse
from pathlib import Path
import cv2

def main():
    a=argparse.ArgumentParser();a.add_argument('--video',required=True);a.add_argument('--out',required=True)
    a.add_argument('--every',type=int,default=30);a.add_argument('--prefix',default='sample')
    x=a.parse_args();out=Path(x.out);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(x.video)
    if not cap.isOpened():raise SystemExit('Cannot open video')
    n=saved=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if n%x.every==0:
            cv2.imwrite(str(out/f'{x.prefix}_{n:07d}.jpg'),frame);saved+=1
        n+=1
    cap.release();print('Saved',saved,'frames to',out)
if __name__=='__main__':main()
