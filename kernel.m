function [trainMatrix testMatrix] = kernel(trainvec,testvec,typeNum)
%each row as a point
trainPoint = size(trainvec,1);
testPoint = size(testvec,1);
dim = size(trainvec,2);

trainMatrix = zeros(trainPoint);
testMatrix = zeros(testPoint,trainPoint);

%% RBF kernel
if typeNum == 1
    for i=1:trainPoint
        for j=i:trainPoint
            trainMatrix(i,j) = sum((trainvec(i,:)-trainvec(j,:)).^2);
            trainMatrix(j,i) = trainMatrix(i,j);
        end
    end
    for i=1:testPoint
        for j=1:trainPoint
            testMatrix(i,j) = sum((testvec(i,:)-trainvec(j,:)).^2);
        end
    end
    gamma = sum(sum(trainMatrix));
    gamma = gamma/trainPoint/trainPoint;
    gamma = 1/gamma;
    trainMatrix = exp(-gamma*trainMatrix);
    testMatrix = exp(-gamma*testMatrix);
end

%% linear
if typeNum == 2
    for i=1:trainPoint
        for j=i:trainPoint
            trainMatrix(i,j) = sum(trainvec(i,:).*trainvec(j,:));
            trainMatrix(j,i) = trainMatrix(i,j);
        end
    end

    for i=1:testPoint
        for j=1:trainPoint
            testMatrix(i,j) = sum(testvec(i,:).*trainvec(j,:));
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
            trainMatrix(i,j) = sum(d3);
            trainMatrix(j,i) = trainMatrix(i,j);
        end
    end
    for i=1:testPoint
        for j=1:trainPoint
            d1 = testvec(i,:)-trainvec(j,:);
            d2 = testvec(i,:)+trainvec(j,:);
            d3 = (d1.^2)./(d2+0.00000000001);
            testMatrix(i,j) = sum(d3);
        end
    end
    gamma = sum(sum(trainMatrix));
    gamma = gamma/trainPoint/trainPoint;
    gamma = 1/gamma;
    trainMatrix = exp(-gamma*trainMatrix);
    testMatrix = exp(-gamma*testMatrix);
end

%% Histogram Intersection
if typeNum == 4
    for i=1:trainPoint
        for j=i:trainPoint
            HItmp = min(trainvec(i,:),trainvec(j,:));
            trainMatrix(i,j) = sum(HItmp);
            trainMatrix(j,i) = trainMatrix(i,j);
        end
    end

    for i=1:testPoint
        for j=1:trainPoint
            HItmp = min(testvec(i,:),trainvec(j,:));
            testMatrix(i,j) = sum(HItmp);
        end
    end
    gamma = sum(sum(trainMatrix));
    gamma = gamma/trainPoint/trainPoint;
    gamma = 1/gamma;
    trainMatrix = exp(-gamma*trainMatrix);
    testMatrix = exp(-gamma*testMatrix);
end