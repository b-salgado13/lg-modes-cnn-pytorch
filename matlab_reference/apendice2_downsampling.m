function [FieldD]=DownSampling_v2(FieldFinal,SquareSize)
[OutPixel,OutPixel2,x]=size(FieldFinal);
res=mod(OutPixel,SquareSize);
Limit=(OutPixel-res)/SquareSize;

res2=mod(OutPixel2,SquareSize);
Limit2=(OutPixel2-res2)/SquareSize;

    %Down sampling
    for m=1:Limit
        for p=1:Limit2
            ini_m=SquareSize*(m-1)+1;
            fin_m=SquareSize*(m);
            ini_n=SquareSize*(p-1)+1;
            fin_n=SquareSize*(p);
            Im=FieldFinal(ini_m:fin_m,ini_n:fin_n);
            FieldD(m,p)=sum(sum(Im))/(SquareSize^2);
        end
    end
    
end
