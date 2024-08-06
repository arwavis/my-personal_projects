from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/aravindv/Documents/project_database/address_book.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Contacts(db.Model):
    __tablename__ = 'Contacts'
    Contact_ID = db.Column(db.Integer, primary_key=True)
    First_Name = db.Column(db.String)
    Last_Name = db.Column(db.String)
    Email = db.Column(db.String)
    Phone = db.Column(db.String)
    Address = db.Column(db.String)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact', methods=['POST'])
def add_contact():
    data = request.json
    new_contact = Contacts(
        First_Name=data['firstName'],
        Last_Name=data['lastName'],
        Email=data['email'],
        Phone=data['phone'],
        Address=data['address']
    )
    db.session.add(new_contact)
    db.session.commit()
    return jsonify({"message": "Contact added successfully"}), 201


@app.route('/contact/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def contact(id):
    contact = Contacts.query.get(id)
    if request.method == 'GET':
        if contact:
            return jsonify({
                "Contact_ID": contact.Contact_ID,
                "First_Name": contact.First_Name,
                "Last_Name": contact.Last_Name,
                "Email": contact.Email,
                "Phone": contact.Phone,
                "Address": contact.Address
            })
        return jsonify({"message": "Contact not found"}), 404
    elif request.method == 'PUT':
        if contact:
            data = request.json
            contact.First_Name = data['firstName']
            contact.Last_Name = data['lastName']
            contact.Email = data['email']
            contact.Phone = data['phone']
            contact.Address = data['address']
            db.session.commit()
            return jsonify({"message": "Contact updated successfully"})
        return jsonify({"message": "Contact not found"}), 404
    elif request.method == 'DELETE':
        if contact:
            db.session.delete(contact)
            db.session.commit()
            return jsonify({"message": "Contact deleted successfully"})
        return jsonify({"message": "Contact not found"}), 404


if __name__ == '__main__':
    app.run(debug=True)
