pos_ratio = 0.2;
neg_ratio=1-pos_ratio;

num_neg=round(5000*neg_ratio);
num_pos=round(5000*pos_ratio);
num_pos_tr=round(num_pos*2/3);
num_pos_te = num_pos - num_pos_tr;
num_neg_tr=round(num_neg*2/3);
num_neg_te = num_neg - num_neg_tr;

negG=(randn(num_neg,2))*3;

%create data
posG=randn(round(num_pos),2)*[0.5,-0.5;-0.5,0.2]+repmat([2,2],round(num_pos),1);
data=[posG;negG];
train_data=data([1:num_pos_tr,num_pos+1:num_pos+num_neg_tr],:);
test_data=data([num_pos_tr+1:num_pos,num_pos+num_neg_tr+1:num_pos+num_neg],:);

train_class = [ones(num_pos_tr,1); zeros(num_neg_tr,1)];
test_class = [ones(num_pos_te,1); zeros(num_neg_te,1)];
labels = [ones(num_pos_tr + num_pos_te,1); zeros(num_neg_tr + num_neg_te,1)];

[W, b] = hybrid(data, labels, 0.8);


figure;
hold;
plot(data(labels == 0,1),data(labels == 0,2),'bx','MarkerSize', 3);
plot(data(labels == 1,1),data(labels == 1,2),'r*','MarkerSize',2);
axis equal;
title('Hybrid');

xx=(b - W(2) * (-5)) / W(1);
yy=(b - W(2) * ( 5)) / W(1);
if(abs(xx)<7 || abs(yy)<7)
    plot([xx,yy],[-4,4],'k','LineWidth',2);
else
    xx=(b-W(1)*(-7))/W(2);
    yy=(b-W(1)*(7))/W(2);
    plot([-7,7],[xx,yy],'k','LineWidth',2);
end   
