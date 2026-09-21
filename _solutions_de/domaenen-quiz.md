---
title: Domänen-Quiz
description: Überprüfung von Domänenwissen durch gezielte Fragen.
category:
- Communication
- Team
problems:
- knowledge-gaps
- implicit-knowledge
- difficult-developer-onboarding
- incomplete-knowledge
- inconsistent-knowledge-acquisition
layout: solution
lang: de
en_slug: domain-quiz
related_solutions:
- slug: knowledge-sharing-practices
  similarity: 0.7
- slug: knowledge-base
  similarity: 0.7
- slug: domain-experts
  similarity: 0.7
- slug: pair-and-mob-programming
  similarity: 0.7
- slug: subject-matter-reviews
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.65
---

## Description

Ein Domänen-Quiz ist eine strukturierte Menge gezielter Fragen, gemeinsam mit Domänenexperten gestaltet, die das Verständnis der Entwickler für die in einem System eingebetteten Geschäftskonzepte, Regeln und Randfälle prüft — genutzt während des Onboardings oder periodisch mit dem gesamten Team durchgeführt, um Lücken im Domänenwissen aufzudecken, bevor diese Lücken Implementierungsfehler verursachen. Dies ist ein bewusst risikoarmes Format, um implizites Wissen explizit und testbar zu machen: Legacy-Systeme kodieren typischerweise Jahre angehäufter Geschäftsregeln und Eigenheiten, die nie vollständig dokumentiert wurden, sodass das Vertrauen eines Entwicklers in sein eigenes Verständnis kein zuverlässiges Signal dafür ist, ob dieses Verständnis tatsächlich korrekt oder vollständig ist. Ein Quiz durchzuführen deckt genau auf, wo dieses Vertrauen fehlplatziert ist — es offenbart oft, dass selbst erfahrene Teammitglieder Lücken bei spezifischen undokumentierten Regeln oder historischen Entscheidungen tragen —, bevor diese Lücken sich als Preisfehler, falsche Berechnungen oder andere Geschäftslogikfehler in Produktion manifestieren. Weil die Fragen gemeinsam mit Domänenexperten geschrieben werden, fungieren die Quiz-Ergebnisse auch als Diagnose für die Dokumentation selbst und weisen direkt darauf hin, welche Bereiche des tatsächlichen Systemverhaltens am schlechtesten irgendwo schriftlich erfasst sind. Gut eingesetzt, fungiert ein Domänen-Quiz als wiederkehrender Pulscheck auf institutionelles Wissen statt als einmaliges Onboarding-Tor, was am meisten in Systemen zählt, wo dieses Wissen dünn, über das Team ungleich verteilt und in Gefahr ist, mit irgendeiner einzelnen Person zur Tür hinauszugehen.

## How to Apply ◆

- Erstellen Sie Quizze, die das Verständnis der Entwickler für Schlüsselgeschäftskonzepte, Regeln und im Legacy-System implementierte Prozesse prüfen.
- Nutzen Sie Quizze während des Onboardings, um das Domänenwissen neuer Entwickler zu bewerten und Bereiche zu identifizieren, in denen Schulung nötig ist.
- Beziehen Sie Fragen zu legacy-spezifischen Eigenheiten ein: undokumentierte Geschäftsregeln, historische Entscheidungen und bekannte Randfälle.
- Führen Sie periodische Domänen-Quizze mit dem gesamten Team durch, um Wissenslücken aufzudecken, bevor sie Implementierungsfehler verursachen.
- Gestalten Sie Fragen gemeinsam mit Domänenexperten, um sicherzustellen, dass sie echt wichtiges Geschäftswissen widerspiegeln.
- Nutzen Sie Quiz-Ergebnisse, um gezielte Wissensaustausch-Sitzungen und Dokumentationsverbesserungen zu leiten.

## Tradeoffs ⇄

**Vorteile:**
- Offenbart Wissenslücken in einem risikoarmen Format, bevor sie zu Implementierungsfehlern führen.
- Schafft eine strukturierte Baseline zur Bewertung des Domänenverständnisses über das Team hinweg.
- Hebt Bereiche hervor, in denen Legacy-System-Dokumentation fehlt.
- Macht implizites Domänenwissen explizit und testbar.

**Kosten:**
- Die Quiz-Erstellung erfordert Aufwand von Domänenexperten und erfahrenen Entwicklern.
- Quizze können herablassend wirken, wenn sie nicht als Lernwerkzeuge statt als Bewertungen positioniert werden.
- Schriftliche Quizze erfassen möglicherweise nicht das nuancierte Verständnis, das für komplexe Domänenentscheidungen nötig ist.
- Die Pflege des Quiz-Inhalts erfordert Aktualisierungen, während sich Domäne und System weiterentwickeln.

## How It Could Be

Ein Legacy-Frachtmanagementsystem hat komplexe Regeln zur Berechnung von Versandtarifen, die von Spediteurverträgen, Gefahrgutklassifikationen und saisonalen Zuschlägen abhängen. Neue Entwickler führen häufig Preisfehler ein, weil sie diese Domänennuancen nicht verstehen. Das Team erstellt ein Domänen-Quiz, das die zwanzig am häufigsten missverstandenen Geschäftsregeln abdeckt, einschließlich Fragen wie „Was passiert mit dem Basistarif, wenn eine Sendung eine Zonengrenze während eines saisonalen Zuschlagzeitraums überschreitet?" Neue Entwickler machen das Quiz in ihrer zweiten Woche, und Ergebnisse werden in einer Nachfolgesitzung mit einem erfahrenen Entwickler besprochen. Das Quiz offenbart, dass selbst erfahrene Teammitglieder Lücken in ihrem Verständnis der Gefahrgutklassifikationsregeln haben, was eine fokussierte Wissensaustausch-Sitzung auslöst, die eine Fehlerklasse verhindert, die vierteljährlich wiederkehrte.
