import subprocess

'''Module with functions for making movies.
NPM 2016'''


def make_movie(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False, save_into_subdir=False):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    Parameters
    ----------
    imgname : str
        path and filename for the images to turn into a movie
    movname : str
        path and filename for output movie
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    imgdir : str or None
        folder to delete if rm_images and save_into_subdir are both True, ie folder containing the images
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    """
    subprocess.call(
        ['./ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    # Delete the original images
    if rm_images:
        print 'Deleting the original images...'
        if save_into_subdir and imgdir is not None:
            print 'Deleting folder ' + imgdir
            subprocess.call(['rm', '-r', imgdir])
        else:
            print 'Deleting folder contents ' + imgdir + imgname + '*.png'
            subprocess.call(['rm', '-r', imgdir + imgname + '*.png'])
