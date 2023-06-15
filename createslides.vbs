Attribute VB_Name = "Module1"
Option Explicit

Public Function CreateSlides()

    Dim pptApp As PowerPoint.Application
    Dim pptPresentation As PowerPoint.Presentation
    Dim pptSlide As PowerPoint.Slide
    
    'Create a new PowerPoint application
    Set pptApp = New PowerPoint.Application
    
    'Create a new presentation
    Set pptPresentation = pptApp.Presentations.Add
    
    'Create a function to create a slide
    Public Function CreateSlide(slideIndex As Integer, slideLayout As PowerPoint.PpSlideLayout) As PowerPoint.Slide
    
        Dim pptSlide As PowerPoint.Slide
        
        Set pptSlide = pptPresentation.Slides.Add(slideIndex, slideLayout)
        
        Set CreateSlide = pptSlide
    
    End Function
    
    'Create the five slides
    For slideIndex = 1 To 5
    
        'Create a slide with the title only layout
        Set pptSlide = CreateSlide(slideIndex, PowerPoint.PpSlideLayout.ppLayoutTitleOnly)
    
        'Set the title of the slide
        pptSlide.Shapes.Title.TextFrame.TextRange.Text = "Slide " & slideIndex
    
        Next slideIndex
    
    End Function
    
    'Call the CreateSlides function to create the five slides
    CreateSlides
    
    'Show the PowerPoint application
    pptApp.Visible = True
    
    'Clean up
    Set pptSlide = Nothing
    Set pptPresentation = Nothing
    Set pptApp = Nothing

End Function

