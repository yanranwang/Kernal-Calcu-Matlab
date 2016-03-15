function [kernelMatrix] = kernelSingNoExp(trainvec,typeNum)
%each row as a point
trainPoint = size(trainvec,1);
dim = size(trainvec,2);

kernelMatrix = zeros(trainPoint);

%% RBF kernel
if typeNum == 1
    for i = 1 : trainPoint
        for j=i:trainPoint
            kernelMatrix(i,j) = sum((trainvec(i,:) - trainvec(j,:)).^2);
            kernelMatrix(j,i) = kernelMatrix(i,j);
        end
    end
    
end

%% linear
if typeNum == 2
    for i=1:trainPoint
        for j=i:trainPoint
            kernelMatrix(i,j) = sum(trainvec(i,:).*trainvec(j,:));
            kernelMatrix(j,i) = kernelMatrix(i,j);
        end
    end

end

%% chi square kernel
if typeNum == 3
    for i=1:trainPoint
        for j=i:trainPoint
            d1 = trainvec(i,:)-trainvec(j,:);
            d2 = trainvec(i,:)+trainvec(j,:);
            d3 = (d1.^2)./(d2+0.00000000001);
            kernelMatrix(i,j) = sum(d3);
            kernelMatrix(j,i) = kernelMatrix(i,j);
        end
    end
end

%% Histogram Intersection
if typeNum == 4
    for i=1:trainPoint
        for j=i:trainPoint
            HItmp = min(trainvec(i,:),trainvec(j,:));
            kernelMatrix(i,j) = sum(HItmp);
            kernelMatrix(j,i) = kernelMatrix(i,j);
        end
    end

end