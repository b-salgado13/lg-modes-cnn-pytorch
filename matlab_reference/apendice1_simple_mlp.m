DownDiv=20;%Tamaño del cuadrado para el downsampling
class=19;%Número de clases en las que se clasifican las imágenes

%Leer las imagenes .jpg en los directorios imgPath de los modos de luz
images=[];
for i=-9:9
    imgPath ='/Volumes/INGFISICA/Red Neuronal Modos Espaciales de Luz/Fotos/m__';
    imgType='/*.jpg';
    images=[images; dir([imgPath num2str(i) imgType])];
end
[ren,col]=size(images);
Target=zeros(class,ren); %Se inicializa el target

for k=1:ren
    %Extraemos el número de la imágen y folder
    aux=str2double(regexp(images(k).name,'\d+[\.]?\d*','match'));
    C(k)=aux(end);
    aux2=regexp(images(k).folder,'\_','split');
    i=aux2{end};
    %Leer imagen
    NameImages=[imgPath,i,'\',images(k).name];
    I=imread(NameImages);
    Im = im2gray(I);%Escala de grises
    Icropped=imcrop(Im,[1900 1100 2899 2499]);%Recorta los bordes negros que no aportan información
    Id=double(Icropped);%Cambio a valores numéricos para el downsampling
    I_down=DownSampling_v2(Id,DownDiv);%Función de Down Sampling en archivo adjunto
    [ren1,col1]=size(I_down);
    Data(:,k)=reshape(I_down,[ren1*col1,1]);%Guardar la imágen k como un solo vector de entrada para la red
    Target(str2num(i)+10,k)=1;%Cada imagen corresponde a un modo espacial que va del -9 al 10, pero los targets se enumeran del 1 al 18
end
 
%% Neural network
Net = patternnet(5);
Net.divideFcn = 'dividerand';
Net.divideMode = 'sample'; 
Net.divideParam.trainRatio=0.7;
Net.divideParam.valRatio=0.15;
Net.divideParam.testRatio = 0.15;
NetTrained = train(Net,Data,Target);
genFunction(NetTrained,'NN_ideal','MatrixOnly','yes')
