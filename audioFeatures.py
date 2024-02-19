%matplotlib inline
import librosa
import librosa.display
import IPython
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
class AudioFeatureExtraction:

    def __init__(self,y,sr,name):
        self.y = y
        self.sr = sr
        self.name = name

    def hp(self):
        self.y_harmonic, self.y_percussive = librosa.effects.hpss(self.y)
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(self.y_harmonic, sr=self.sr, alpha=0.25)
        librosa.display.waveshow(self.y_percussive, sr=self.sr, color='r', alpha=0.5)
        plt.title(f'Harmonic + Percussive of sample{self.name}')

    def tempoBeat(self):
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr)
        print('Detected Tempo of '+self.name+': '+str(self.tempo)+ ' beats/min')
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sr)
        self.beat_time_diff=np.ediff1d(self.beat_times)
        self.beat_nums = np.arange(1, np.size(self.beat_times))
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 5)
        ax.set_ylabel("Time difference (s)")
        ax.set_xlabel("Beats")
        g=sns.barplot(x = self.beat_nums,y = self.beat_time_diff, palette="BuGn_d",ax=ax)
        g=g.set(xticklabels=[])

    def chroma(self):
        self.chroma=librosa.feature.chroma_cens(y=self.y_harmonic, sr=self.sr)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(self.chroma,y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chroma of sample{self.name}')

    def mfcc(self):
        self.mfccs = librosa.feature.mfcc(y=self.y_harmonic, sr=self.sr, n_mfcc=13)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(self.mfccs, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC of {self.name}')

    def spectral_centroid(self):
        self.cent = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 1, 1)
        plt.semilogy(self.cent.T, label=f'Spectral centroid of {self.name}')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, self.cent.shape[-1]])
        plt.legend()

    def spectral_contrast(self):
        self.contrast=librosa.feature.spectral_contrast(y=self.y_harmonic,sr=self.sr)
        plt.figure(figsize=(15,5))
        librosa.display.specshow(self.contrast, x_axis='time')
        plt.colorbar()
        plt.ylabel('Frequency bands')
        plt.title(f'Spectral contrast of {self.name}')

    def roll_off(self):
        self.rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        plt.figure(figsize=(15,5))
        plt.semilogy(self.rolloff.T, label=f'Roll-off frequency of {self.name}')
        plt.ylabel('Hz')
        plt.xticks([])
        plt.xlim([0, self.rolloff.shape[-1]])
        plt.legend()

    def zcr(self):
        self.zrate=librosa.feature.zero_crossing_rate(self.y_harmonic)
        plt.figure(figsize=(15,5))
        plt.semilogy(self.zrate.T, label='Fraction')
        plt.ylabel('Fraction per Frame')
        plt.xticks([])
        plt.xlim([0, self.rolloff.shape[-1]])
        plt.legend()
        plt.title(f'Zero Crossing Rate of {self.name}')

    def chroma_mean_std(self) :
        self.chroma_mean=np.mean(self.chroma,axis=1)
        self.chroma_std=np.std(self.chroma,axis=1)
        #plot the summary
        octave=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        plt.figure(figsize=(15,5))
        plt.title(f'Mean CENS of {self.name}')
        sns.barplot(x=octave,y=self.chroma_mean)

        plt.figure(figsize=(15,5))
        plt.title(f'SD CENS of {self.name}')
        sns.barplot(x=octave,y=self.chroma_std)

        #Generate the chroma Dataframe
        self.chroma_df=pd.DataFrame()
        for i in range(0,12):
            self.chroma_df['chroma_mean_'+str(i)]=self.chroma_mean[i]
        for i in range(0,12):
            self.chroma_df['chroma_std_'+str(i)]=self.chroma_mean[i]
        self.chroma_df.loc[0]=np.concatenate((self.chroma_mean,self.chroma_std),axis=0)
        print(f'Chroma Dataframe of {self.name}')
        print(self.chroma_df)
        print()

    def mfcc_mean_std(self):
        self.mfccs_mean=np.mean(self.mfccs,axis=1)
        self.mfccs_std=np.std(self.mfccs,axis=1)

        self.coeffs=np.arange(0,13)
        plt.figure(figsize=(15,5))
        plt.title(f'Mean MFCCs of {self.name}')
        sns.barplot(x=self.coeffs,y=self.mfccs_mean)

        plt.figure(figsize=(15,5))
        plt.title(f'SD MFCCs of {self.name}')
        sns.barplot(x=self.coeffs,y=self.mfccs_std)
        #Generate the chroma Dataframe
        self.mfccs_df=pd.DataFrame()
        for i in range(0,13):
            self.mfccs_df['mfccs_mean_'+str(i)]=self.mfccs_mean[i]
        for i in range(0,13):
            self.mfccs_df['mfccs_std_'+str(i)]=self.mfccs_mean[i]
        self.mfccs_df.loc[0]=np.concatenate((self.mfccs_mean,self.mfccs_std),axis=0)
        print(f'MFCC DataFrame of {self.name}')
        print(self.mfccs_df)
        print()

    def cent_metrices(self):
        self.cent_mean=np.mean(self.cent)
        self.cent_std=np.std(self.cent)
        self.cent_skew=scipy.stats.skew(self.cent,axis=1)[0]
        print(f'Spectral Centroid of {self.name}')
        print('Mean: '+str(self.cent_mean))
        print('SD: '+str(self.cent_std))
        print('Skewness: '+str(self.cent_skew))
        print()

    def contrast_mean_std(self):
        self.contrast_mean=np.mean(self.contrast,axis=1)
        self.contrast_std=np.std(self.contrast,axis=1)
        self.conts=np.arange(0,7)
        plt.figure(figsize=(15,5))
        plt.title('Mean Spectral Contrast')
        sns.barplot(x=self.conts,y=self.contrast_mean)
        plt.figure(figsize=(15,5))
        plt.title('SD Spectral Contrast')
        sns.barplot(x=self.conts,y=self.contrast_std)
        #Generate the chroma Dataframe
        self.contrast_df=pd.DataFrame()
        for i in range(0,12):
            self.chroma_df['chroma_mean_'+str(i)]=self.chroma_mean[i]
        for i in range(0,12):
            self.chroma_df['chroma_std_'+str(i)]=self.chroma_mean[i]
        self.chroma_df.loc[0]=np.concatenate((self.chroma_mean,self.chroma_std),axis=0)
        print(f'Chroma Dataframe of {self.name}')
        print(self.chroma_df)
        print()

    def rolloff_metrices(self):
        self.rolloff_mean=np.mean(self.rolloff)
        self.rolloff_std=np.std(self.rolloff)
        self.rolloff_skew=scipy.stats.skew(self.rolloff,axis=1)[0]
        print(f'Roll Off Metrices of {self.name}')
        print('Mean: '+str(self.rolloff_mean))
        print('SD: '+str(self.rolloff_std))
        print('Skewness: '+str(self.rolloff_skew))
        print()
    def spectralDf(self):
        self.spectral_df=pd.DataFrame()
        self.collist=['cent_mean','cent_std','cent_skew']
        for i in range(0,7):
            self.collist.append('contrast_mean_'+str(i))
        for i in range(0,7):
            self.collist.append('contrast_std_'+str(i))
        self.collist=self.collist+['rolloff_mean','rolloff_std','rolloff_skew']
        for c in self.collist:
            self.spectral_df[c]=0
        self.data=np.concatenate(([self.cent_mean,self.cent_std,self.cent_skew],self.contrast_mean,self.contrast_std,[self.rolloff_mean,self.rolloff_std,self.rolloff_std]),axis=0)
        self.spectral_df.loc[0]=self.data
        print(f'Spectral Dataframe of {self.name}')
        print(self.spectral_df )
        print()

    def zcr_metrices(self):
        self.zrate_mean=np.mean(self.zrate)
        self.zrate_std=np.std(self.zrate)
        self.zrate_skew=scipy.stats.skew(self.zrate,axis=1)[0]
        print(f'ZCR Metrices of {self.name}')
        print('Mean: '+str(self.zrate_mean))
        print('SD: '+str(self.zrate_std))
        print('Skewness: '+str(self.zrate_skew))
        print()

    def zcr_df(self):
        self.zrate_df=pd.DataFrame()
        self.zrate_df['zrate_mean']=0
        self.zrate_df['zrate_std']=0
        self.zrate_df['zrate_skew']=0
        self.zrate_df.loc[0]=[self.zrate_mean,self.zrate_std,self.zrate_skew]
        print(f'Zcr dataframe of {self.name}')
        print(self.zrate_df)
        print()

    def beat_df(self):
        self.beat_df=pd.DataFrame()
        self.beat_df['tempo']=self.tempo
        self.beat_df.loc[0]=self.tempo
        print(f'Beat Dataframe of {self.name}')
        print(self.beat_df)
        print()

    def final_df(self):
        self.final_df=pd.concat((self.chroma_df,self.mfccs_df,self.spectral_df,self.zrate_df,self.beat_df),axis=1)
        print(f'Final Dataframe of {self.name}')
        print(self.final_df.head())
        print()
def splitSamples(y,sr,seconds = 30):
    split = 22050*seconds
    name =1
    samples = []
    for i in range(0,len(y),split):
        if i+split <= len(y):
            samples.append(AudioFeatureExtraction(y[i:i+split+1],sr,"Sample"+str(name)))
        else:
            samples.append(AudioFeatureExtraction(y[i:],sr,"Sample"+str(name)))
        name += 1
    return samples
audio = "YOUR/AUDIO/PATH"
y,sr=librosa.load(audio)
print('Audio Sampling Rate: '+str(sr)+' samples/sec')
print('Total Samples: '+str(np.size(y)))
secs=np.size(y)/sr
print('Audio Length: '+str(secs)+' s')
IPython.display.Audio(audio)
duration = 30 # duration in seconds
samples = splitSamples(y,sr,duration)

'''

# Functions Names:

*  hp()
*  tempoBeat()
*  chroma()
*  mfcc()
*  spectral_centroid()
*  spectral_contrast()
*  roll_off()
*  zcr()
*  chroma_mean_std()
*  mfcc_mean_std()
*  cent_metrices()
*  contrast_mean_std()
*  rolloff_metrices()
*  spectralDf()
*  zcr_metrices()
*  zcr_df()
*  beat_df()
*  final_df()

'''
