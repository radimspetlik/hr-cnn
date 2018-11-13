#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

import os
import shutil
import bob.io.base
import bob.io.base.test_utils
import bob.db.base
import tempfile
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base


regenerate_database = False
dbfile = bob.io.base.test_utils.datafile("test_db.sql3", "bob.db.base")
Base = declarative_base()


class TestFile (Base, bob.db.base.File):
    __tablename__ = "file"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, unique=True)
    path = Column(String(100), unique=True)

    def __init__(self):
        bob.db.base.File.__init__(self, path="test/path")
        self.client_id = 5


def create_database():
    if os.path.exists(dbfile):
        os.remove(dbfile)
    import bob.db.base.utils
    engine = bob.db.base.utils.create_engine_try_nolock(
        'sqlite', dbfile, echo=True)
    Base.metadata.create_all(engine)
    session = bob.db.base.utils.session('sqlite', dbfile, echo=True)
    session.add(TestFile())
    session.commit()
    session.close()
    del session
    del engine


class TestDatabase (bob.db.base.SQLiteDatabase):

    def __init__(self):
        super(TestDatabase, self).__init__(dbfile, TestFile, None, None)

    def groups(self, protocol=None):
        return ['group']

    def model_ids(self, groups=None, protocol=None):
        return [5]

    def objects(self, groups=None, protocol=None, purposes=None,
                model_ids=None):
        return list(self.query(TestFile))

    def tmodel_ids(self, groups=None, protocol=None):
        return self.model_ids()

    def tobjects(self, groups=None, protocol=None, model_ids=None):
        return self.objects()

    def zobjects(self, groups=None, protocol=None):
        return self.objects()

    def annotations(self, file):
        assert isinstance(file, TestFile)
        return {'key': (42, 180)}


def test01_annotations():
    # tests the annotation IO functionality provided by this utility class

    # check the different annotation types
    for annotation_type in ('eyecenter', 'named', 'idiap'):
        # get the annotation file name
        annotation_file = bob.io.base.test_utils.datafile(
            "%s.pos" % annotation_type, 'bob.db.base')
        # read the annotations
        annotations = bob.db.base.read_annotation_file(
            annotation_file, annotation_type)
        # check
        assert 'leye' in annotations
        assert 'reye' in annotations
        assert annotations['leye'] == (20, 40)
        assert annotations['reye'] == (20, 10)
        if annotation_type == 'named':
            assert 'pose' in annotations
            assert annotations['pose'] == 30
        if annotation_type == 'idiap':
            assert 'gender' in annotations
            assert annotations['gender'] == ['M']


def test02_database():
    # check that the database API works
    if regenerate_database:
        create_database()

    db = TestDatabase()

    def check_file(fs):
        assert len(fs) == 1
        f = fs[0]
        assert isinstance(f, TestFile)
        assert f.id == 1
        assert f.client_id == 5
        assert f.path == "test/path"

    check_file(db.objects())
    check_file(db.tobjects())
    check_file(db.zobjects())
    check_file(db.all_files())
    check_file(db.files([1]))
    check_file(db.reverse(["test/path"]))

    model_ids = db.model_ids()
    assert len(model_ids) == 1
    assert model_ids[0] == 5
    tmodel_ids = db.tmodel_ids()
    assert len(tmodel_ids) == 1
    assert tmodel_ids[0] == 5

    file = db.objects()[0]
    assert db.paths([1], "another/directory",
                    ".other")[0] == "another/directory/test/path.other"

    annots = db.annotations(file)
    assert len(annots) == 1
    assert 'key' in annots.keys()
    assert (42, 180) in annots.values()

    # try file save
    temp_dir = tempfile.mkdtemp(prefix="bob_db_test_")
    data = [1., 2., 3.]
    file.save(data, temp_dir)
    assert os.path.exists(file.make_path(temp_dir, ".hdf5"))
    read_data = bob.io.base.load(file.make_path(temp_dir, ".hdf5"))
    for i in range(3):
        assert data[i] == read_data[i]
    shutil.rmtree(temp_dir)

    # check closing and re-opening database connection
    del db
    db = TestDatabase()
    check_file(db.objects())
