import torch
import torch.nn.functional as F

def split_feature_map_into_blocks(feature_map, block_size):
    _, _, height, width = feature_map.size()

    # 检查特征图大小是否小于块的大小，如果是则返回原特征图
    if height < block_size or width < block_size:
        return [feature_map]

    # 计算块的数量
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size

    # 计算填充的大小
    padding_height = block_size - (height % block_size)
    padding_width = block_size - (width % block_size)

    # 对特征图进行填充
    padded_feature_map = F.pad(feature_map, (0, padding_width, 0, padding_height))

    # 切分特征图成块
    blocks = []
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            block = padded_feature_map[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            blocks.append(block)

    # 在通道上进行拼接
    concatenated_blocks = torch.cat(blocks, dim=1)

    return concatenated_blocks


def reconstruct_blocks(blocks, original_size, block_size):
    _, num_channels, _, _ = original_size
    num_blocks_height = original_size[2] // block_size
    num_blocks_width = original_size[3] // block_size

    # 创建一个与原始特征图相同大小的张量来存储复原后的特征图
    reconstructed_feature_map = torch.zeros(*original_size)

    # 将块复原到对应的位置
    block_index = 0
    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            block = blocks[:, block_index]
            reconstructed_feature_map[:, :, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block.unsqueeze(1)
            block_index += 1

    return reconstructed_feature_map


## 创建输入特征图
#feature_map = torch.randn(1, 64, 129, 130)
#
## 设置块的大小
#block_size = 32
#
## 将特征图切分并拼接成块
#concatenated_blocks = split_feature_map_into_blocks(feature_map, block_size)
#
## 打印拼接后的块的形状
#print(concatenated_blocks.shape)
#
#
#print("------------------------------------------------------------------------------")
#
#
## 假设原始图像的大小为 original_size，块的大小为 block_size
#original_size = (1, 64, 129, 130)
#
## 将切割后的块复原为原始图像
#reconstructed_image = reconstruct_blocks(concatenated_blocks, original_size, block_size)
#
## 打印复原后的图像的形状
#print(reconstructed_image.shape)
