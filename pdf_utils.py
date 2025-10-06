import logging
import os
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult

# Optional: set logging level globally
logging.basicConfig(level=logging.INFO)

# --- CONFIG --- 
PDF_SERVICES_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID")
PDF_SERVICES_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET")

def convert_pdf_to_docx_adobe(input_pdf_path, output_docx_path):
    """
    Converts a PDF to DOCX using Adobe PDF Services API.

    Parameters:
        input_pdf_path (str): Path to the input .pdf file.
        output_docx_path (str): Path to write the output .docx file.

    Raises:
        RuntimeError if conversion fails.
    """
    try:
        # Read PDF binary
        with open(input_pdf_path, 'rb') as file:
            input_stream = file.read()

        # Create credentials instance
        credentials = ServicePrincipalCredentials(
            
            client_id=PDF_SERVICES_CLIENT_ID,
            client_secret=PDF_SERVICES_CLIENT_SECRET
        )

        # Create service and upload input
        pdf_services = PDFServices(credentials=credentials)
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

        # Set export parameters and create job
        export_pdf_params = ExportPDFParams(target_format=ExportPDFTargetFormat.DOCX)
        export_pdf_job = ExportPDFJob(input_asset=input_asset, export_pdf_params=export_pdf_params)

        # Submit job and fetch result
        location = pdf_services.submit(export_pdf_job)
        job_result = pdf_services.get_job_result(location, ExportPDFResult)
        result_asset: CloudAsset = job_result.get_result().get_asset()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)

        # Write to DOCX
        with open(output_docx_path, "wb") as output_file:
            output_file.write(stream_asset.get_input_stream())

        print(f"✅ PDF successfully converted to DOCX: {output_docx_path}")
        return output_docx_path

    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logging.exception("❌ Adobe PDF to DOCX conversion failed.")
        raise RuntimeError("Adobe conversion failed") from e
