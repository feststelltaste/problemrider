---
title: Personas
description: Charakterisierung repräsentativer Nutzertypen durch fiktive
  Personen.
category:
- Requirements
problems:
- poor-user-experience-ux-design
- inadequate-requirements-gathering
- requirements-ambiguity
- misaligned-deliverables
- customer-dissatisfaction
- user-frustration
- user-confusion
- feature-bloat
layout: solution
lang: de
en_slug: personas
related_solutions:
- slug: user-stories
  similarity: 0.8
- slug: prototypes
  similarity: 0.75
- slug: on-site-customer
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
- slug: story-mapping
  similarity: 0.75
- slug: prototyping
  similarity: 0.75
---

## Description

Eine Persona ist ein fiktives, aber evidenzbasiertes Komposit einer echten Nutzergruppe — ihrer Ziele, ihres technischen Kenntnisstands, ihrer Frustrationen und der Aufgaben, die sie tatsächlich ausführen —, aufgebaut aus Interviews und, wo verfügbar, Nutzungsanalysen statt aus Annahmen. In der Legacy-Modernisierung dienen Personas einem spezifischen und wertvollen Zweck: Sie hindern das Team daran, standardmäßig jedes bestehende Feature nachzubauen, nur weil es im alten System existierte, indem sie eine explizite Antwort darauf erzwingen, welcher Nutzertyp tatsächlich welche Fähigkeit braucht. Der Bezug auf drei bis fünf gut recherchierte Personas während Priorisierung und Design-Review hält Diskussionen an echten Nutzerbedürfnissen statt an technischer Präferenz verankert, obwohl die Praxis nur hilft, wenn die Personas auf echter Forschung basieren — auf Annahmen aufgebaute Personas können ein Team ebenso sicher fehlleiten wie gar keine Personas.

## How to Apply ◆

> In der Legacy-Modernisierung helfen Personas Teams zu verstehen, wer das alte System tatsächlich nutzt und was diese Nutzer brauchen, statt einfach jedes bestehende Feature zu replizieren.

- Interviewen Sie tatsächliche Nutzer des Legacy-Systems, um unterschiedliche Nutzergruppen mit verschiedenen Zielen, technischen Kenntnisständen und Nutzungsmustern zu identifizieren.
- Erstellen Sie drei bis fünf Personas, die die primären Nutzertypen repräsentieren, und geben Sie jeder einen Namen, eine Rollenbeschreibung, Ziele, Frustrationen mit dem aktuellen System und zentrale Aufgaben, die sie ausführen.
- Validieren Sie Personas gegen Nutzungsanalysen des Legacy-Systems, falls verfügbar — Log-Daten offenbaren oft Nutzerverhaltensmuster, die Interviews übersehen.
- Verwenden Sie Personas während der Feature-Priorisierung, um zu bestimmen, welche Legacy-Features für welche Nutzertypen kritisch sind, und vermeiden Sie die Falle, alles nachzubauen, „weil es da war".
- Beziehen Sie sich in Design-Reviews und Sprint-Planung auf Personas, um Diskussionen auf Nutzerbedürfnisse statt technische Präferenzen fokussiert zu halten.
- Aktualisieren Sie Personas, während die Modernisierung fortschreitet und neues Nutzer-Feedback verfügbar wird.

## Tradeoffs ⇄

> Personas liefern ein gemeinsames Vokabular zur Diskussion von Nutzerbedürfnissen, können aber übervereinfachen, wenn sie nicht in echter Nutzerforschung verankert sind.

**Vorteile:**

- Verhindert Feature-Aufblähung während der Modernisierung, indem klare Kriterien dafür geliefert werden, was jeder Nutzertyp tatsächlich braucht, im Gegensatz zu dem, was das Legacy-System zufällig anbot.
- Schafft Empathie für Endnutzer innerhalb des Entwicklungsteams, besonders wenn Teammitglieder das Legacy-System nie selbst genutzt haben.
- Hilft, Modernisierungsanstrengungen zu priorisieren, indem identifiziert wird, welche Nutzergruppen am meisten von Legacy-System-Beschränkungen betroffen sind.
- Bietet einen gemeinsamen Bezugspunkt zur Lösung von Meinungsverschiedenheiten über Feature-Umfang und Designentscheidungen.

**Kosten und Risiken:**

- Schlecht recherchierte, auf Annahmen statt echten Nutzerdaten basierende Personas können das Team in die Irre führen und ein falsches Verständnisgefühl erzeugen.
- Personas können veralten, wenn sie nicht aktualisiert werden, während die Modernisierung Nutzer-Arbeitsabläufe und -erwartungen ändert.
- Teams könnten für eine Persona übermäßig optimieren auf Kosten anderer, wenn Persona-Prioritäten nicht gut ausbalanciert sind.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Personas Legacy-System-Modernisierungsentscheidungen leiten.

Eine Universität, die ihr Legacy-Studierendenregistrierungssystem ersetzte, erstellte vier Personas: einen Erstsemesterstudenten, der mit dem Prozess nicht vertraut ist, einen Studenten im letzten Jahr, der sich zum letzten Mal registriert, einen akademischen Berater, der Hunderte von Studenten verwaltet, und einen Registrar-Administrator, der Ausnahmen handhabt. Das Legacy-System behandelte all diese Nutzer identisch und präsentierte ihnen dieselbe komplexe Oberfläche. Durch das Design des Ersatzes um personaspezifische Arbeitsabläufe konnte das Team die studentenseitige Erfahrung dramatisch vereinfachen, während die Power-User-Features erhalten blieben, auf die sich Berater und Administratoren verließen. Features, die nur von Administratoren genutzt wurden, wurden hinter rollenbasiertem Zugang verschoben, statt die Oberfläche jedes Nutzers zu überladen.
