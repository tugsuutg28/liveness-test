"""Evaluate saved model on held-out test images. Reports confusion matrix."""
import argparse, json
from pathlib import Path

def main(kind,threshold):
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
    from ml.pipeline import load_classifier
    root=Path(__file__).resolve().parents[1]
    model=load_classifier(kind,root)
    if model is None: raise SystemExit('Train the model first.')
    ds=tf.keras.utils.image_dataset_from_directory(root/'datasets'/kind/'test',
        class_names=['real','fake'],image_size=(160,160),batch_size=16,shuffle=False)
    y=[];probs=[]
    for x,labels in ds:
        y.extend(labels.numpy().tolist());probs.extend(model.predict(x,verbose=0).reshape(-1).tolist())
    pred=(np.array(probs)>=threshold).astype(int)
    report={'threshold':threshold,'labels':['real','fake'],
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
        'classification_report':classification_report(y,pred,target_names=['real','fake'],zero_division=0,output_dict=True),
        'roc_auc':float(roc_auc_score(y,probs)) if len(set(y))==2 else None}
    out=root/'reports'/f'{kind}_evaluation.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--kind',choices=['liveness','deepfake'],required=True)
    a.add_argument('--threshold',type=float,default=.5);args=a.parse_args();main(args.kind,args.threshold)
