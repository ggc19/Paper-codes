%%  Establish the time-frequency spectrogram
clear;clc;
Data_Iron=xlsread('Read_Data_Iron_1.xlsx');       
Data_MDPE=xlsread('Read_Data_MDPE_1.xlsx');  
sample1=size(Data_Iron,1);  
sample2=size(Data_MDPE,1);  
samples = sample1+sample2;
Data = zeros(sample2, 4801);
for i=1:sample1
    Data(i,:) = Data_Iron(i,:);
end
for i=1:sample2
    Data(i+sample1,:) = Data_MDPE(i,:);  
end

Fs=4800; id=0; 
wlen=256; inc=64; win=hanning(wlen);   % different resolution corresponds to different 'wlen' and 'inc', 
for i =1: samples
    temp = Data(i+id, :);
    y = enframe(temp(2:end), win, inc)';    %  the data is framed
    fn = size(y,2);                                      %  fn is the number of frame
    w2=wlen/2+1; n2=40 : w2-50;   % Data1: n2=10 : w2-20£»Data2: n2=20 : w2-30£»Data3: n2=40 : w2-50£»
    frameTime = (((1:fn)-1)*inc+wlen/2)/Fs;
    freq = (n2-1)*Fs/wlen; Y=fft(y);
    Data1(i,:,:) = abs(Y(n2,:));
    Data1_y(i,1) = temp(1) ;
    % generate WGN
    snr1=[5,10]; 
     for j=1:length(snr1)  
        [new_temp, noise]=Gnoisegen(temp(2:end), snr1(j));     
        fp=[100 2000]; fs=[50 2050]; rp=2; rs=5; wp=fp*2*pi/Fs; ws=fs*2*pi/Fs;
        [n,wn]=buttord(wp/pi,ws/pi,rp,rs); [bz,az]=butter(n,wp/pi);
        new_temp = filter(bz, az,new_temp); 
        y = enframe(new_temp, win, inc)';
        fn = size(y,2);
        frameTime = (((1:fn)-1)*inc+wlen/2)/Fs;
        freq = (n2-1)*Fs/wlen; Y=fft(y);
        Data1_noise1(j+length(snr1)*(i-1),:,:) = abs(Y(n2,:));
        Data1_noise1_y(j+length(snr1)*(i-1),1)=temp(1);
    end
    snr2=[-10,-5, 0];
    for j=1:length(snr2)  
        [new_temp, noise]=Gnoisegen(temp(2:end), snr2(j));     
        fp=[100 2000]; fs=[50 2050]; rp=2; rs=5; wp=fp*2*pi/Fs; ws=fs*2*pi/Fs;
        [n,wn]=buttord(wp/pi,ws/pi,rp,rs); [bz,az]=butter(n,wp/pi);
        new_temp = filter(bz, az,new_temp); 
        y = enframe(new_temp, win, inc)';
        fn = size(y,2);
        frameTime = (((1:fn)-1)*inc+wlen/2)/Fs;
        freq = (n2-1)*Fs/wlen; Y=fft(y);
        Data1_noise2(j+length(snr2)*(i-1),:,:) = abs(Y(n2,:));
        Data1_noise2_y(j+length(snr2)*(i-1),1)=temp(1);
    end
end
save Data1.mat Data1;  
save Data1_y.mat Data1_y;
save Data1_noise1.mat Data1_noise1;  
save Data1_noise1_y.mat Data1_noise1_y;
save Data1_noise2.mat Data1_noise2;  
save Data1_noise2_y.mat Data1_noise2_y;

%% Extract features 
Fs = 4800;  wlen = 4600;
df = Fs/wlen; km = floor(wlen/8);   
fx1 = fix(100/df)+1;  fx2 = fix(2400/df)+1;  
k=0.5; aparam=2; 
Feature = zeros(samples*3, 14);  % 14 features
for i=1: samples*3
    Leak1 = Data1(i, 2:end);      
    A=abs(fft(Leak1));
    E=zeros(wlen/2+1,1); 
    E(fx1+1:fx2-1)=A(fx1+1:fx2-1);
    E1 = E.*E;
    P1=E1/sum(E1);
    index=find(P1>=0.9);
    if ~isempty(index), E(index)=0; E1(index)=0; end
    for m=1:km   
        Eb(m)=sum(E1(4*m-3 : 4*m));
    end
    prob=(Eb+k)/sum(Eb+k);                   
    Hb(i)=-sum(prob.*log(prob+eps));      
    Esum(i)=log10(1+sum(E1)/aparam);  
    prob = E/(sum(E));                            
    H(i) = -sum(prob.*log(prob+eps));      
    Ef(i) = sqrt(1 + abs(Esum(i)/H(i)));      
    tz=steager(Leak1);                                            
    En(i,1)=mean(tz); En(i,2)=mean(abs(Leak1));  
    En(i,3)=rms(abs(Leak1));                               
    En(i,4)=mean(20*log10(A(fx1+1:fx2-1)));      
    Zcr(i)=sum(Leak1(1:end-1).*Leak1(2:end)<0);       
    auto_coef = xcorr(Leak1,'coeff');                  
    coeff = sum(auto_coef(wlen-5 : wlen+5).^2)/(sum(auto_coef(:).^2));   
    % EMD, obtain the RMS and Shannon entropy of IMFs 1-3 
    imp=emd(Leak1);
    for j = 1 :3                                    
        En(i,4+j) = rms(abs(imp(j,:)));        
        Ee = abs(imp(j,:)).^2;                                
        prob = Ee./sum(Ee);                                 
        H_imf = -sum(prob.*log(prob+eps));       
        En(i,7+j) =H_imf;
    end
    Feature(i,1)=Hb(i); Feature(i,2)=Ef(i); Feature(i,3)=En(i,1); Feature(i,4)=En(i,2); Feature(i,5)=En(i,3); Feature(i,6)=En(i,4); Feature(i,7)=Zcr(i); Feature(i,8)=coeff;
    Feature(i,9:14) = En(i, 5:10); 
end 
save Feature.mat Feature;  
