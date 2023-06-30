function appendedimg = appendimages(img1, img2)

rows1 = size(img1,1);
rows2 = size(img2,1);

if (rows1 < rows2)
     img1(rows2,1) = 0;
else
     img2(rows1,1) = 0;
end

appendedimg = [img1 img2];   
end